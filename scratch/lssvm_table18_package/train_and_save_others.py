"""Exporta XGBoost, FT-Softmax e FT-CUR (streaming) para medir o embarque na MESMA CPU dos LSSVMs (2026-10-15).

Mesmas bases e mesmo split de train_and_save.py (N = 20 mil, teste de 2000), hiperparâmetros vencedores
do Tier 2 (config/tier2_fixed_params.json). Treino CURTO (10 épocas) na CPU: o objetivo é medir tamanho
e custo de predição, que dependem só da arquitetura; a F1 que vale é a publicada (Ablação D).

Formatos de embarque:
  XGBoost   <base>_xgb.ubj         modelo nativo do XGBoost
  FT-*      <base>_ftsoftmax.pt    TorchScript (scaler embutido)
  FT-CUR    <base>_ftcur.pt        TorchScript em modo streaming: encoder + K_m e resumo congelados
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "../.."); sys.path.insert(0, "../../scripts"); sys.path.insert(0, ".")
import run_table18_benchmark as B
import run_tier2_gridcv as R2

R2.BUDGET_OVERRIDE.update(epochs=10, patience=10)
OUT = Path("saved_models")
CFG = json.load(open("../../config/tier2_fixed_params.json"))
DS = {"adult": "ADULT", "credit": "CREDIT", "bank": "BANK"}


class FTExport(torch.nn.Module):
    def __init__(self, net, mu, s):
        super().__init__(); self.net = net
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.mu) / self.s)


class FTCURStream(torch.nn.Module):
    """Mesma conta de FTTransformerCURColnorm._logits_streaming, com o resumo do treino congelado."""
    def __init__(self, clf, mu, s):
        super().__init__(); u = clf._model; a = u.inter_attn
        self.u, self.q_proj, self.out_proj, self.norm, self.head = u, a.q_proj, a.out_proj, a.norm, u.head
        self.h, self.dh, self.dm, self.scale = a.n_heads, a.d_head, a.d_model, float(a.scale)
        self.register_buffer("K_m", clf._stream_K_m.detach().clone())
        self.register_buffer("S", clf._stream_summary.detach().clone())
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))

    def forward(self, x):
        cls = self.u.get_cls_embeddings((x - self.mu) / self.s)
        c = cls.shape[0]
        Q = self.q_proj(cls).view(c, self.h, self.dh).permute(1, 0, 2)
        C = torch.softmax(Q @ self.K_m.transpose(-1, -2) / self.scale, dim=-1)
        out = (C @ self.S).permute(1, 0, 2).reshape(c, self.dm)
        return self.head(self.norm(self.out_proj(out) + cls)).squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = []
    for ds, DSU in DS.items():
        os.environ["DATASET"] = ds
        Xtr, Xte, ytr, yte = B.make_data(20000, 0)
        ytr01, yte01 = (ytr + 1) // 2, (yte + 1) // 2
        Xte32 = torch.tensor(Xte, dtype=torch.float32)
        for name, variant in (("xgb", "XGBoost"), ("ftsoftmax", "FTTransformer_softmax"), ("ftcur", "FTTransformerCURRandom")):
            pipe, _ = R2._build_pipeline(variant, 0)
            pipe.set_params(**{f"clf__{k}": v for k, v in CFG[variant][DSU].items()})
            t = time.time(); pipe.fit(Xtr, ytr01); fit_s = time.time() - t
            sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
            if name == "xgb":
                f = OUT / f"{ds}_xgb.ubj"; clf._model.save_model(f)
                ref = pipe.predict(Xte)
                import xgboost as xgb
                bst = xgb.Booster(); bst.load_model(f)
                same = bool(np.array_equal((bst.inplace_predict(sc.transform(Xte)) >= 0.5).astype(int), ref))
                extra = dict(n_trees=int(CFG[variant][DSU]["n_estimators"]), max_depth=int(CFG[variant][DSU]["max_depth"]))
            else:
                if name == "ftsoftmax":
                    mod = FTExport(clf.model_.cpu().eval(), sc.mean_, sc.scale_)
                    ref = pipe.predict(Xte)
                else:
                    clf._precompute_train_summary(); clf.predict_mode = "streaming"
                    mod = FTCURStream(clf, sc.mean_, sc.scale_).eval()
                    ref = pipe.predict(Xte)
                with torch.no_grad():
                    ts = torch.jit.trace(mod, Xte32[:4])
                f = OUT / f"{ds}_{name}.pt"; ts.save(str(f))
                with torch.no_grad():
                    same = bool(np.array_equal((torch.jit.load(str(f))(Xte32).numpy() >= 0).astype(int), np.asarray(ref)))
                extra = dict(n_params=int(sum(p.numel() for p in mod.parameters())),
                             n_buffers=int(sum(b.numel() for b in mod.buffers())))
            rows.append(dict(base=ds, model=name, fit_s=round(fit_s, 1), file_kb=round(f.stat().st_size / 1024, 1),
                             export_pred_equal=same, **extra))
            print(rows[-1], flush=True)
            json.dump(rows, open(OUT / "summary_others.json", "w"), indent=2)
