#!/usr/bin/env python3
"""Esparsidade NOS PESOS do FT-Transformer (o que ocupa os KB), comparável à do ADMM nos α.

FT softmax denso (arquitetura fixa: 3 blocos, 4 cabeças, d = 64; protocolo do Tier 1: hold-out 70/30,
lote cheio, 200 épocas) → poda por MAGNITUDE global, não estruturada, de todos os pesos de matriz
(tokenizer, W_Q/W_K/W_V/W_O, FFN, cabeça; LayerNorm, vieses e CLS ficam) → ajuste fino com a máscara
fixa (50 épocas, mesmo otimizador) → teste. Esparsidades s ∈ {0, 0,5, 0,75, 0,9, 0,95, 0,98, 0,99}.

KB: denso = 4 B × parâmetros; esparso = valores não nulos (4 B) + índice (2 B) por não nulo, mais os
parâmetros não podados (formato COO/CSR simples, como o embarque guardaria). A referência do ADMM-Nyström
vem do Tier 1 (n_support_vectors × (d + 1) floats + viés).

Braço CUR/Nyström: cada Linear vira C·(U⁺R) com r colunas/linhas sorteadas, fatorado em duas Linear
(o ajuste fino mantém o posto); r/min(out, in) ∈ {0,5, 0,25, 0,125, 0,0625}.

    .venv/bin/python scripts/study_ft_weight_pruning.py
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import run_tier2_gridcv as g  # noqa: E402
from src.data.preprocessing import make_splits, _convert_labels  # noqa: E402
from src.models.transformers.ft_transformer import FTTransformer  # noqa: E402

ARCH = {"num_blocks": 3, "num_heads": 4, "embedding_dim": 64, "dropout": 0.1, "lr": 1e-3,
        "batch_size": 4096, "max_epochs": 200, "patience": 200, "attention_type": "softmax"}
LEVELS = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
RANKS = [0.5, 0.25, 0.125, 0.0625]     # r / min(saída, entrada) no braço CUR
FINETUNE_EPOCHS = 50


def _factor(W: torch.Tensor, r: int, mode: str, rng: np.random.Generator, idx=None):
    """W (out × in) ≈ second · first, com first: r × in e second: out × r.
    cur/cur_shared: C·(U⁺R) com r colunas e r linhas (idx = (linhas, colunas) dados ou sorteados);
    svd: o melhor posto r (Eckart–Young), U_r·(S_r V_rᵀ) — o teto da compressão de posto."""
    out_f, in_f = W.shape
    if mode == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return (S[:r, None] * Vh[:r]), U[:, :r].clone()
    rows, cols = idx if idx is not None else (rng.choice(out_f, r, replace=False), rng.choice(in_f, r, replace=False))
    rows = torch.as_tensor(rows, device=W.device); cols = torch.as_tensor(cols, device=W.device)
    return torch.linalg.pinv(W[rows][:, cols]) @ W[rows, :], W[:, cols].clone()


def cur_factorize(net: nn.Module, rank_frac: float, rng: np.random.Generator, mode: str = "cur") -> None:
    """Cada nn.Linear W (out × in) vira Linear(in→r) ∘ Linear(r→out); guarda r·(out + in).
    mode="cur": índices sorteados por matriz (uniforme, como o Nyström do estudo).
    mode="cur_shared": dentro de cada atenção, W_Q, W_K e W_V usam as MESMAS linhas e colunas
                       (o produto Q·Kᵀ combina dimensões de mesmo índice); o resto como "cur".
    mode="svd": SVD truncado em cada matriz."""
    shared = {}
    if mode == "cur_shared":
        for att in net.modules():
            if all(hasattr(att, a) for a in ("q_proj", "k_proj", "v_proj")):
                out_f, in_f = att.q_proj.weight.shape
                r = max(1, int(round(rank_frac * min(out_f, in_f))))
                idx = (rng.choice(out_f, r, replace=False), rng.choice(in_f, r, replace=False))
                for a in ("q_proj", "k_proj", "v_proj"):
                    shared[id(getattr(att, a))] = idx
    for parent in list(net.modules()):
        for name, lin in list(parent.named_children()):
            if not isinstance(lin, nn.Linear):
                continue
            W = lin.weight.data; out_f, in_f = W.shape
            r = max(1, int(round(rank_frac * min(out_f, in_f))))
            if r * (out_f + in_f) >= out_f * in_f:
                continue                                   # fatorar não economiza: fica densa
            w1, w2 = _factor(W, r, "svd" if mode == "svd" else "cur", rng, shared.get(id(lin)))
            first = nn.Linear(in_f, r, bias=False).to(W.device); first.weight.data = w1
            second = nn.Linear(r, out_f, bias=lin.bias is not None).to(W.device); second.weight.data = w2
            if lin.bias is not None:
                second.bias.data = lin.bias.data.clone()
            setattr(parent, name, nn.Sequential(first, second))


def prunable(net: nn.Module):
    """(módulo, 'weight') de toda matriz de pesos: Linear + tokenizer (LayerNorm/vieses/CLS ficam)."""
    out = [(m, "weight") for m in net.modules() if isinstance(m, nn.Linear)]
    out.append((net.tokenizer, "weight"))
    return out


def n_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def size_kb(net: nn.Module) -> tuple[float, float, int, int]:
    pruned_ids = {id(getattr(m, "weight")) for m, _ in prunable(net)}
    total = nnz = rest = 0
    for m, _ in prunable(net):
        w = m.weight
        total += w.numel(); nnz += int((w != 0).sum())
    for p in net.parameters():
        if id(p) not in pruned_ids and p.ndim and not any(p is getattr(m, "weight_orig", None) for m, _ in prunable(net)):
            rest += p.numel()
    n_all = total + rest
    return 4 * n_all / 1024, (6 * nnz + 4 * rest) / 1024, nnz, n_all


def finetune(clf: FTTransformer, X: np.ndarray, y: np.ndarray, epochs: int):
    net = clf.model_; dev = next(net.parameters()).device
    lab = {c: float(i) for i, c in enumerate(clf.classes_)}
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    y_t = torch.tensor([lab[c] for c in y], dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=clf.lr, weight_decay=clf.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(); net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_t), device=dev)
        for s in range(0, len(X_t), clf.batch_size):
            b = perm[s:s + clf.batch_size]
            opt.zero_grad(); loss_fn(net(X_t[b]), y_t[b]).backward(); opt.step()
    net.eval()


def admm_ref(ds: str, seed: int):
    recs = json.load(open(ROOT / "results/tier1_gridcv.json"))
    r = next((r for r in recs if r["dataset"] == ds and r["model"] == "ADMMNystromLSSVM" and r["seed"] == seed), None)
    if r is None:
        return None
    kb = 4 * (r["n_support_vectors"] * (r["n_features"] + 1) + 1) / 1024
    return {"admm_f1": r["test_f1_macro"], "admm_kb": kb, "admm_sparsity": r["sparsity_ratio"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["AI4I", "PID", "BCW", "AUS"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--levels", nargs="+", type=float, default=LEVELS)
    p.add_argument("--ranks", nargs="+", type=float, default=RANKS)
    p.add_argument("--lowrank", nargs="+", default=["cur"], help="cur, cur_shared, svd")
    p.add_argument("--output", default=str(ROOT / "results/study_ft_weight_pruning.json"))
    a = p.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["dataset"], r["seed"]) for r in recs}
    for ds in a.datasets:
        X, y, _ = g.DatasetLoader.load(ds)
        for seed in a.seeds:
            if (ds, seed) in done:
                continue
            X_tr, X_te, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            ytr, yte = _convert_labels(ytr, "binary"), _convert_labels(yte, "binary")
            sc = StandardScaler().fit(X_tr); X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
            g.set_global_seed(seed)
            t0 = time.perf_counter()
            dense = FTTransformer(random_state=seed, **ARCH).fit(X_tr, ytr)
            ref = admm_ref(ds, seed)
            for s in a.levels:
                clf = copy.deepcopy(dense)
                if s > 0:
                    prune.global_unstructured(prunable(clf.model_), pruning_method=prune.L1Unstructured, amount=s)
                    f1_0 = float(f1_score(yte, clf.predict(X_te), average="macro"))   # antes do ajuste fino
                    torch.manual_seed(seed); finetune(clf, X_tr, ytr, FINETUNE_EPOCHS)
                    for m, n in prunable(clf.model_):
                        prune.remove(m, n)
                else:
                    f1_0 = None
                f1 = float(f1_score(yte, clf.predict(X_te), average="macro"))
                kb_dense, kb_sparse, nnz, n_all = size_kb(clf.model_)
                rec = {"dataset": ds, "seed": seed, "sparsity": s, "test_f1_macro": f1, "f1_before_finetune": f1_0,
                       "nnz_weights": nnz, "n_params": n_all, "kb_dense": kb_dense, "kb_sparse": kb_sparse,
                       **(ref or {})}
                rec["method"] = "magnitude"
                recs.append(rec)
                print(f"{ds:5s} s{seed} esparsidade {s:.2f}  F1 {f1:.4f}"
                      + (f" (sem ajuste fino {f1_0:.4f})" if f1_0 is not None else "")
                      + f"  {min(kb_dense, kb_sparse):7.1f} KB"
                      + (f"   | ADMM F1 {ref['admm_f1']:.4f} {ref['admm_kb']:.1f} KB" if ref else ""), flush=True)
            n_dense = n_params(dense.model_)
            for meth, rf in [(m_, r_) for m_ in a.lowrank for r_ in a.ranks]:   # braços de posto baixo
                clf = copy.deepcopy(dense)
                cur_factorize(clf.model_, rf, np.random.default_rng(seed), meth)
                f1_0 = float(f1_score(yte, clf.predict(X_te), average="macro"))
                torch.manual_seed(seed); finetune(clf, X_tr, ytr, FINETUNE_EPOCHS)
                f1 = float(f1_score(yte, clf.predict(X_te), average="macro"))
                n_p = n_params(clf.model_)
                rec = {"dataset": ds, "seed": seed, "method": meth, "rank_frac": rf, "test_f1_macro": f1,
                       "f1_before_finetune": f1_0, "n_params": n_p, "kb_dense": 4 * n_p / 1024,
                       "kb_sparse": 4 * n_p / 1024, "sparsity": 1 - n_p / n_dense, **(ref or {})}
                recs.append(rec)
                print(f"{ds:5s} s{seed} {meth:10s} r/min={rf:.4f}  compressão {rec['sparsity']:.2f}  F1 {f1:.4f}"
                      f" (sem ajuste fino {f1_0:.4f})  {rec['kb_dense']:7.1f} KB", flush=True)
            out.write_text(json.dumps(recs, indent=1))
            print(f"   ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
