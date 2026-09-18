#!/usr/bin/env python3
"""Benchmark completo para atualizar a Tabela 19 da dissertação.

IMPORTANTE sobre o FT-CUR: o `_logits` publicado até aqui tem um bug
ESTRUTURAL (não uma opção de design) -- o resumo da atenção Nyström usa
[X_train || X_test] inteiro na predição, então a predição de um ponto de
teste muda dependendo de quais OUTROS pontos de teste estão no mesmo lote,
e a memória de predição cresce sem limite com N (OOM verificado em
N=50000). O bug independe de como o TREINO é batchado (mini ou full) --
por isso a correção se aplica às DUAS configurações de FT-CUR que a
Tabela 19 já tem, sem exceção: não faz sentido corrigir uma e deixar a
outra com um bug estrutural conhecido. `FTCUR_minibatch` e
`FTCUR_mfixed_full` aqui já SÃO as versões corrigidas -- não é uma
alternativa entre opções igualmente válidas. O comportamento antigo é
mantido só como `*_bug_reference`, rodado unicamente para documentar a
magnitude do bug (evidência do "antes"), não como candidato a entrar na
tabela final.

Modelos:
    FTTransformer_softmax, FTTransformer_topk, FTTransformer_entmax,
    FTTransformer_sparsemax        — baselines de atenção inter-atributos
    SAINT_minibatch, SAINT_fullbatch
    FTCUR_minibatch                — CORRIGIDO (resumo travado no treino +
        softmax online, streaming; batch_size=256 no treino). É o número
        que deve ir pra tabela.
    FTCUR_mfixed_full              — CORRIGIDO, igual ao minibatch, mas com
        batch_size=None no treino (preserva o regime full-batch que essa
        variante testa; o chunk_size do precompute/predição cai pro default
        1024 sozinho, via fallback `self.batch_size or 1024`).
    FTCUR_minibatch_bug_reference,
    FTCUR_mfixed_full_bug_reference — o `_logits` ANTIGO, com o bug, nas
        mesmas duas configs. Existem só para medir o "antes" e dar
        evidência da correção; não são variantes, são o erro documentado.

Cada (modelo, N) é medido com FIT e PREDIÇÃO separados — a métrica que a
Tabela 19 publicada nunca teve para a predição (só existia para o fit).

Uso (rodar de dentro do repo clonado, ou apontando --repo-root):
    python run_table19_benchmark.py --output results_table19.json

O script acha a raiz do repo sozinho (procura um diretório com src/models/
ft_transformer_model.py subindo a partir de onde está); se não achar, passe
--repo-root explicitamente.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("table19_bench")


# ── Achar a raiz do repo (funciona não importa onde o script for colocado) ──

def _find_repo_root(start: Path, marker: str = "src/models/ft_transformer_model.py") -> Path:
    cur = start.resolve()
    for _ in range(6):
        if (cur / marker).exists():
            return cur
        cur = cur.parent
    raise SystemExit(
        f"Não achei a raiz do repo (procurei por '{marker}' subindo a partir de {start}). "
        "Rode com --repo-root apontando para a pasta que contém src/."
    )


def _setup_paths(repo_root: str | None) -> None:
    root = Path(repo_root) if repo_root else _find_repo_root(Path(__file__).parent)
    sys.path.insert(0, str(root))
    log.info("Repo root: %s", root)


# ── Configuração ──────────────────────────────────────────────────────────────

N_VALUES  = [500, 1000, 2000, 5000, 10000, 20000, 50000]
N_TEST    = 500
REPEATS   = 3
EPOCHS    = 40
PATIENCE  = 6

BATCH = 256
M_FIXED = 256
M_MINIBATCH = 64

N_MAX = {v: 50000 for v in (
    "FTTransformer_softmax", "FTTransformer_topk", "FTTransformer_entmax",
    "FTTransformer_sparsemax", "SAINT_minibatch", "SAINT_fullbatch",
    "FTCUR_minibatch", "FTCUR_mfixed_full",
    "FTCUR_minibatch_bug_reference", "FTCUR_mfixed_full_bug_reference",
)}

_PROC = psutil.Process(os.getpid())


# ── Correção streaming (pontos 1+2): resumo travado no treino ──────────────
#
# O FTTransformerCURColnorm original monta o resumo (R, W, R@V) da atenção
# Nyström usando K,V de TODO o contexto concatenado [X_train || X_test] na
# predição -- não só do treino. Duas consequências: (1) a predição de um
# ponto de teste muda dependendo de quais OUTROS pontos de teste estão no
# mesmo lote (acoplamento espúrio, verificado com diff exato = 0 após a
# correção); (2) como não há chunking nesse passo, a memória de predição
# cresce com N e estoura em N grande (verificado: OOM em N=50000 aqui).
#
# A correção trava o resumo usando SÓ X_train (softmax online, streaming,
# nunca materializa o contexto inteiro de uma vez) -- aí cada predição fica
# genuinamente independente das outras, e a memória de predição fica O(chunk),
# não O(N).

def _register_streaming_ftcur():
    from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
    from src.models.ft_transformer_model import _truncated_pinv

    @torch.no_grad()
    def _precompute_train_summary(self, chunk_size: int):
        underlying = self._model
        attn = underlying.inter_attn
        idx = self._landmark_idx_full_
        X_train_np = self.X_train_
        n_tr = len(X_train_np)
        H, dh = attn.n_heads, attn.d_head

        X_lm_t = self._to_tensor(X_train_np[idx.cpu().numpy()])
        cls_lm = underlying.get_cls_embeddings(X_lm_t)
        m = cls_lm.shape[0]
        Q_m = attn.q_proj(cls_lm).view(m, H, dh).permute(1, 0, 2)
        K_m = attn.k_proj(cls_lm).view(m, H, dh).permute(1, 0, 2)
        W = torch.softmax(Q_m @ K_m.transpose(-1, -2) / attn.scale, dim=-1)
        U_inv = torch.stack([_truncated_pinv(W[h], attn.tau_ratio) for h in range(H)], dim=0)

        running_max = torch.full((H, m), float("-inf"), device=Q_m.device)
        running_sum = torch.zeros((H, m), device=Q_m.device)
        running_out = torch.zeros((H, m, dh), device=Q_m.device)
        for s in range(0, n_tr, chunk_size):
            X_chunk_t = self._to_tensor(X_train_np[s:s + chunk_size])
            cls_chunk = underlying.get_cls_embeddings(X_chunk_t)
            c = cls_chunk.shape[0]
            K_chunk = attn.k_proj(cls_chunk).view(c, H, dh).permute(1, 0, 2)
            V_chunk = attn.v_proj(cls_chunk).view(c, H, dh).permute(1, 0, 2)
            scores = Q_m @ K_chunk.transpose(-1, -2) / attn.scale
            chunk_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, chunk_max)
            correction = torch.exp(running_max - new_max)
            p = torch.exp(scores - new_max.unsqueeze(-1))
            running_sum = running_sum * correction + p.sum(dim=-1)
            running_out = running_out * correction.unsqueeze(-1) + p @ V_chunk
            running_max = new_max
        RV = running_out / running_sum.unsqueeze(-1)
        return K_m, U_inv @ RV

    class FTTransformerCURColnormStreaming(FTTransformerCURColnorm):
        def fit(self, X, y):
            super().fit(X, y)
            bs = self.batch_size or 1024
            self._stream_K_m, self._stream_summary = _precompute_train_summary(self, bs)
            return self

        @torch.no_grad()
        def _logits(self, X):
            underlying = self._model
            attn = underlying.inter_attn
            bs = self.batch_size or 1024
            parts = []
            for s in range(0, len(X), bs):
                X_chunk_t = self._to_tensor(X[s:s + bs])
                cls = underlying.get_cls_embeddings(X_chunk_t)
                c = cls.shape[0]
                Q = attn.q_proj(cls).view(c, attn.n_heads, attn.d_head).permute(1, 0, 2)
                C = torch.softmax(Q @ self._stream_K_m.transpose(-1, -2) / attn.scale, dim=-1)
                out = C @ self._stream_summary
                out = out.permute(1, 0, 2).reshape(c, attn.d_model)
                out = attn.out_proj(out)
                out = attn.norm(out + cls)
                logits = underlying.head(out).squeeze(-1)
                parts.append(logits.cpu().numpy())
            return np.concatenate(parts)

    return FTTransformerCURColnormStreaming


# ── Instanciação dos modelos ──────────────────────────────────────────────────

def _make_model(variant: str):
    if variant in ("FTTransformer_softmax", "FTTransformer_topk",
                   "FTTransformer_entmax", "FTTransformer_sparsemax"):
        from src.models.transformers.ft_transformer import FTTransformer
        attn_map = {
            "FTTransformer_softmax":   "softmax",
            "FTTransformer_topk":      "topk",
            "FTTransformer_entmax":    "entmax",
            "FTTransformer_sparsemax": "sparsemax",
        }
        return FTTransformer(
            num_blocks=2, num_heads=2,
            max_epochs=EPOCHS, patience=PATIENCE,
            attention_type=attn_map[variant],
        )
    elif variant in ("SAINT_minibatch", "SAINT_fullbatch"):
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        bs = BATCH if variant == "SAINT_minibatch" else None
        return SAINTColnorm(
            n_heads=2, n_layers=1,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
            batch_size=bs,
        )
    elif variant == "FTCUR_minibatch":
        # CORRIGIDO: mesma config de sempre (m=64, batch_size=256), mas com
        # o _logits streaming -- este é o número que vai pra tabela final.
        cls = _register_streaming_ftcur()
        return cls(
            n_heads=4, n_layers=2, m_landmarks=M_MINIBATCH, batch_size=BATCH,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
        )
    elif variant == "FTCUR_minibatch_bug_reference":
        # O _logits ANTIGO (com o bug), mesma config de FTCUR_minibatch --
        # só para medir o "antes" e documentar a magnitude do problema.
        # NÃO é candidato a entrar na tabela final.
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        return FTTransformerCURColnorm(
            n_heads=4, n_layers=2, m_landmarks=M_MINIBATCH, batch_size=BATCH,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
        )
    elif variant == "FTCUR_mfixed_full":
        # CORRIGIDO, igual ao minibatch -- o bug é estrutural, não uma opção
        # de design, então não faz sentido corrigir um e deixar o outro.
        # batch_size=None preserva o treino full-batch (é o que "full"
        # significa); o fallback `self.batch_size or 1024` dentro da classe
        # streaming cuida do chunk_size do precompute/predição sozinho --
        # não precisa (nem deve) forçar batch_size aqui.
        cls = _register_streaming_ftcur()
        return cls(
            n_heads=4, n_layers=2, m_landmarks=M_FIXED, batch_size=None,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
        )
    elif variant == "FTCUR_mfixed_full_bug_reference":
        # O _logits ANTIGO, mesma config de FTCUR_mfixed_full -- só
        # evidência do "antes", não candidato a entrar na tabela final.
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        return FTTransformerCURColnorm(
            n_heads=4, n_layers=2, m_landmarks=M_FIXED, batch_size=None,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
        )
    raise ValueError(variant)


# ── Monitoramento de memória ──────────────────────────────────────────────────

def _monitor_rss(stop: threading.Event, peak_mb: list) -> None:
    while not stop.is_set():
        try:
            rss = _PROC.memory_info().rss / 1024 / 1024
            if rss > peak_mb[0]:
                peak_mb[0] = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01)


def _run_phase(fn):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline_mb = _PROC.memory_info().rss / 1024 / 1024
    peak_mb = [baseline_mb]
    stop = threading.Event()
    mon = threading.Thread(target=_monitor_rss, args=(stop, peak_mb), daemon=True)
    mon.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    stop.set()
    mon.join()
    ram_delta = max(peak_mb[0] - baseline_mb, 0.0)
    vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
               if torch.cuda.is_available() else 0.0)
    return elapsed, ram_delta, vram_mb


def _measure(variant: str, X_tr, y_tr, X_te) -> dict:
    model = _make_model(variant)
    fit_s, fit_ram, fit_vram = _run_phase(lambda: model.fit(X_tr, y_tr))
    pred_oom = False
    try:
        pred_s, pred_ram, pred_vram = _run_phase(lambda: model.predict(X_te))
        pred_ms = pred_s * 1000
    except torch.cuda.OutOfMemoryError:
        pred_oom = True
        pred_ms = pred_ram = pred_vram = float("nan")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "fit_s": round(fit_s, 3),
        "ram_delta_mb": round(fit_ram, 1),
        "vram_mb": round(fit_vram, 1),
        "pred_oom": pred_oom,
        "pred_ms": None if pred_oom else round(pred_ms, 3),
        "pred_ram_mb": None if pred_oom else round(pred_ram, 1),
        "pred_vram_mb": None if pred_oom else round(pred_vram, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="table19_results.json")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--repo-root", default=None,
                        help="Pasta que contém src/ (default: acha sozinho)")
    parser.add_argument("--variants", default=None,
                        help="Lista separada por vírgula p/ rodar só um subconjunto. Default: todas.")
    args = parser.parse_args()

    _setup_paths(args.repo_root)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    # Resumível: sessões do Kaggle podem cair no meio. Se já existe saída
    # parcial, carrega e pula tudo que já está lá (medido OU decidido-como-
    # skip, incluindo os que foram pulados em cascata após um OOM).
    results: list[dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done_keys = {(r["variant"], r["n"]) for r in results}
    if results:
        log.info("Retomando de %s: %d combinações já registradas.", out_path, len(done_keys))

    variants = [
        "FTTransformer_softmax", "FTTransformer_topk",
        "FTTransformer_entmax", "FTTransformer_sparsemax",
        "SAINT_minibatch", "SAINT_fullbatch",
        "FTCUR_minibatch",                    # CORRIGIDO -- é o número final
        "FTCUR_mfixed_full",                  # CORRIGIDO -- idem
        "FTCUR_minibatch_bug_reference",      # o "antes" -- só evidência, não vai pra tabela
        "FTCUR_mfixed_full_bug_reference",    # idem
    ]
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in variants]
        if unknown:
            raise SystemExit(f"Variantes desconhecidas: {unknown}\nDisponíveis: {variants}")
        variants = wanted
        log.info("Rodando apenas: %s", variants)

    for variant in variants:
        log.info("=== %s ===", variant)
        max_n = N_MAX[variant]

        for n in N_VALUES:
            if (variant, n) in done_keys:
                continue

            if n > max_n:
                results.append({"variant": variant, "n": n, "skipped": True})
                done_keys.add((variant, n))
                continue

            fits, preds, rams, vrams = [], [], [], []
            pred_rams, pred_vrams = [], []
            pred_oom = False
            oom = False
            last_error = None

            for rep in range(args.repeats):
                try:
                    X_all, y_all = make_classification(
                        n_samples=n + N_TEST,
                        n_features=20, n_informative=10, n_redundant=5,
                        random_state=rep,
                    )
                    X_tr_raw, X_te_raw = X_all[:n], X_all[n:]
                    y_tr = y_all[:n]
                    X_tr = scaler.fit_transform(X_tr_raw)
                    X_te = scaler.transform(X_te_raw)

                    m = _measure(variant, X_tr, y_tr, X_te)
                    fits.append(m["fit_s"]); rams.append(m["ram_delta_mb"]); vrams.append(m["vram_mb"])
                    if m["pred_oom"]:
                        pred_oom = True
                    else:
                        preds.append(m["pred_ms"]); pred_rams.append(m["pred_ram_mb"]); pred_vrams.append(m["pred_vram_mb"])

                    log.info("  N=%6d rep=%d  fit=%.1fs  pred=%s  fitVRAM=%.1fMB  predVRAM=%s",
                             n, rep, m["fit_s"],
                             "OOM" if m["pred_oom"] else f"{m['pred_ms']:.1f}ms",
                             m["vram_mb"],
                             "OOM" if m["pred_oom"] else f"{m['pred_vram_mb']:.1f}MB")
                except torch.cuda.OutOfMemoryError:
                    oom = True; last_error = "OOM"
                    log.warning("  N=%6d rep=%d OOM (fit) — pulando N maiores", n, rep)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                except Exception as e:
                    last_error = str(e)
                    log.warning("  N=%6d rep=%d ERRO: %s", n, rep, e)
                    break

            if not fits:
                results.append({"variant": variant, "n": n, "skipped": True, "oom": oom, "error": last_error})
                done_keys.add((variant, n))
                if oom:
                    for n_skip in N_VALUES[N_VALUES.index(n) + 1:]:
                        results.append({"variant": variant, "n": n_skip, "skipped": True,
                                        "oom": True, "error": "OOM (skip após primeiro OOM)"})
                        done_keys.add((variant, n_skip))
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2)
                    break
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                continue

            have_pred = len(preds) > 0
            done_keys.add((variant, n))
            results.append({
                "variant": variant, "n": n, "skipped": False,
                "fit_s_median":      round(float(np.median(fits)), 3),
                "ram_delta_mb_median": round(float(np.median(rams)), 1),
                "vram_mb_median":    round(float(np.median(vrams)), 1),
                "pred_oom":          pred_oom,
                "pred_ms_median":    round(float(np.median(preds)), 3) if have_pred else None,
                "pred_ram_mb_median":  round(float(np.median(pred_rams)), 1) if have_pred else None,
                "pred_vram_mb_median": round(float(np.median(pred_vrams)), 1) if have_pred else None,
            })

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Salvo em %s", out_path)


if __name__ == "__main__":
    main()
