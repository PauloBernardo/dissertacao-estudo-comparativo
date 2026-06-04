"""Generate figures for the experiment report (updated with DualFISTA + honest Nyström)."""
import json, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

OUT = Path('results/report_figs')
OUT.mkdir(parents=True, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
t1_base   = json.load(open('results/tier1_results.json'))
t1_custom = json.load(open('results/tier1_custom_models.json'))
all_t1    = t1_base + t1_custom
scaling   = json.load(open('results/synthetic_scaling_n2000.json'))
noise5    = json.load(open('results/synthetic_5features_tuned.json'))
mk5_data  = json.load(open('results/synthetic_mk5.json'))

DATASETS_T1 = ['BCW','PID','HAB','VCP','GCR','AUS','TWS','TWM','TWC']
SYNTH_DS    = ['TWS','TWM','TWC']

# ── Label / colour mappings ───────────────────────────────────────────────────
MODEL_LABEL = {
    'StandardLSSVM':         'LSSVM-Std',
    'PCPLSSVm':              'LSSVM-PCP',
    'FSALSSVm':              'LSSVM-FSA',
    'IPLSSVm':               'LSSVM-IP',
    'PruningLSSVM':          'LSSVM-Prun',
    'OppositeMapsLSSVM':     'LSSVM-OppM',
    'ADMMNesterovLSSVM':     'LSSVM-ADMM',
    'FTTransformer_softmax': 'FT-Softmax',
    'FTTransformer_topk':    'FT-TopK',
    'FTTransformer_entmax':  'FT-Entmax',
    'FTTransformer_sparsemax':'FT-Sparsemax',
    # proposed / investigated
    'OriginalADMM':          'LSSVM-ADMM*',
    'ADMMElasticNet':        'ADMM-ElasticNet',
    'FISTANesterov':         'FISTA-Nesterov',
    'DualFISTA':             'DualFISTA★',
    'NystromLSSVMColnorm':   'Nyström-SVM★',
    'FTTransformerCURColnorm':'FT-CUR★',
    'SAINTColnorm':          'SAINT',
    'XGBoost':               'XGBoost',
}

# Colour groups
LSSVM_BASELINE = {'StandardLSSVM','PCPLSSVm','FSALSSVm','IPLSSVm',
                  'PruningLSSVM','OppositeMapsLSSVM','FISTANesterov'}
FT_BASELINE    = {'FTTransformer_softmax','FTTransformer_topk',
                  'FTTransformer_entmax','FTTransformer_sparsemax','SAINTColnorm'}
PAPER_BASE     = {'OriginalADMM','ADMMNesterovLSSVM'}
INVESTIGATED   = {'ADMMElasticNet'}
PROPOSED       = {'DualFISTA','NystromLSSVMColnorm','FTTransformerCURColnorm'}
GENERAL_BASE   = {'XGBoost'}

def model_color(m):
    if m in PROPOSED:
        return {'DualFISTA':'#d62728',
                'NystromLSSVMColnorm':'#2ca02c',
                'FTTransformerCURColnorm':'#ff7f0e'}[m]
    if m in GENERAL_BASE:
        return '#e377c2'  # rosa para XGBoost
    if m in INVESTIGATED:
        return '#9467bd'
    if m in PAPER_BASE:
        return '#8c564b'
    if m in LSSVM_BASELINE:
        return '#1f77b4'
    return '#888888'

def collect(records, datasets, model_key=None):
    """Return {model: {dataset: [f1, ...]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get('status') != 'ok':
            continue
        m = r.get('model_variant') or r.get('model')
        if model_key and m != model_key:
            continue
        d = r['dataset']
        if d in datasets:
            out[m][d].append(r.get('f1_macro', float('nan')))
    return out

# ── Deduplicate: use ADMMNesterovLSSVM from base, skip OriginalADMM for Tier1
def dedup_t1(records):
    seen = defaultdict(set)
    out = []
    for r in records:
        m = r.get('model_variant') or r.get('model')
        if m == 'OriginalADMM':
            continue  # redundant with ADMMNesterovLSSVM
        key = (m, r['dataset'], r.get('seed'))
        if key not in seen[m]:
            seen[m].add((r['dataset'], r.get('seed')))
            out.append(r)
    return out

all_t1_dedup = dedup_t1(all_t1)

# ── Figure 1: F1-macro all models Tier1 ──────────────────────────────────────
sc = collect(all_t1_dedup, DATASETS_T1)

# compute mean±std aggregated over all datasets per model
model_stats = {}
for m, ds_map in sc.items():
    all_vals = []
    for vals in ds_map.values():
        all_vals.extend(vals)
    model_stats[m] = (np.mean(all_vals), np.std(all_vals))

# Order: best to worst
order = sorted(model_stats, key=lambda m: -model_stats[m][0])
labels = [MODEL_LABEL.get(m, m) for m in order]
means  = [model_stats[m][0] for m in order]
stds   = [model_stats[m][1] for m in order]
colors = [model_color(m) for m in order]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(order)), means, xerr=stds, color=colors,
               align='center', alpha=0.85, capsize=3, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('F1-macro médio (30 seeds × 9 datasets)', fontsize=10)
ax.set_title('Tier 1 — Desempenho Geral dos Modelos\n(★ = modelo proposto/investigado)', fontsize=11)
ax.axvline(means[0], color='gray', linestyle=':', linewidth=0.8)
ax.set_xlim(0.55, 0.95)

patches = [
    mpatches.Patch(color='#1f77b4', label='LSSVM baselines'),
    mpatches.Patch(color='#888888', label='FT-Transformer baselines'),
    mpatches.Patch(color='#9467bd', label='ADMM family'),
    mpatches.Patch(color='#d62728', label='DualFISTA (proposto)'),
    mpatches.Patch(color='#2ca02c', label='Nyström-SVM (proposto)'),
    mpatches.Patch(color='#ff7f0e', label='FT-CUR (proposto)'),
]
ax.legend(handles=patches, fontsize=8, loc='lower right')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT / 'fig1_f1_all_models.pdf', dpi=150)
plt.savefig(OUT / 'fig1_f1_all_models.png', dpi=150)
plt.close()
print("fig1 done")

# ── Figure 2: Sparsity trade-off ──────────────────────────────────────────────
SPARSE_MODELS = ['ADMMNesterovLSSVM','ADMMElasticNet','FISTANesterov',
                 'DualFISTA','FSALSSVm','IPLSSVm','PruningLSSVM',
                 'OppositeMapsLSSVM','NystromLSSVMColnorm']

spar_stats = {}
for r in all_t1_dedup:
    if r.get('status') != 'ok': continue
    m = r.get('model_variant') or r.get('model')
    if m not in SPARSE_MODELS: continue
    d = r['dataset']
    if d not in DATASETS_T1: continue
    if m not in spar_stats:
        spar_stats[m] = {'f1': [], 'spar': []}
    spar_stats[m]['f1'].append(r.get('f1_macro', float('nan')))
    spar_stats[m]['spar'].append(r.get('sparsity_ratio', 0) * 100)

fig, ax = plt.subplots(figsize=(8, 6))
for m in SPARSE_MODELS:
    if m not in spar_stats: continue
    f1_mean  = np.mean(spar_stats[m]['f1'])
    f1_std   = np.std(spar_stats[m]['f1'])
    sp_mean  = np.mean(spar_stats[m]['spar'])
    sp_std   = np.std(spar_stats[m]['spar'])
    col  = model_color(m)
    lbl  = MODEL_LABEL.get(m, m)
    mrk  = '*' if m in PROPOSED else 'o'
    sz   = 180 if m in PROPOSED else 80
    ax.errorbar(sp_mean, f1_mean, xerr=sp_std, yerr=f1_std,
                fmt=mrk, color=col, markersize=10 if m in PROPOSED else 7,
                capsize=4, linewidth=1.2, zorder=5 if m in PROPOSED else 3)
    offset_y = 0.004 if m not in {'FISTANesterov','ADMMElasticNet'} else -0.009
    ax.annotate(lbl, (sp_mean, f1_mean + offset_y), fontsize=7.5, ha='center')

ax.set_xlabel('Esparsidade média (%)', fontsize=10)
ax.set_ylabel('F1-macro médio', fontsize=10)
ax.set_title('Trade-off Esparsidade × Desempenho\n(★ = modelos propostos)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 100)
plt.tight_layout()
plt.savefig(OUT / 'fig2_sparsity_tradeoff.pdf', dpi=150)
plt.savefig(OUT / 'fig2_sparsity_tradeoff.png', dpi=150)
plt.close()
print("fig2 done")

# ── Figure 3: Scaling N=400 vs N=2000 ────────────────────────────────────────
sc400  = collect(all_t1_dedup, SYNTH_DS)
sc2000 = collect(scaling, ['TWS_2k','TWM_2k','TWC_2k'])

SCALE_MODELS = ['StandardLSSVM','ADMMNesterovLSSVM','DualFISTA',
                'FTTransformer_sparsemax','FTTransformer_softmax',
                'NystromLSSVMColnorm','FTTransformerCURColnorm']
ds_pairs = [('TWS','TWS_2k'), ('TWM','TWM_2k'), ('TWC','TWC_2k')]
ds_names = ['TWS','TWM','TWC']

x = np.arange(len(SCALE_MODELS))
width = 0.12
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for col_idx, (ds400, ds2k) in enumerate(ds_pairs):
    ax = axes[col_idx]
    for i, m in enumerate(SCALE_MODELS):
        v400 = np.mean(sc400[m].get(ds400, [float('nan')])) if sc400[m].get(ds400) else float('nan')
        v2k  = np.mean(sc2000[m].get(ds2k, [float('nan')])) if sc2000[m].get(ds2k) else float('nan')
        col = model_color(m)
        lbl = MODEL_LABEL.get(m, m)
        ax.bar(i - 0.2, v400, 0.35, color=col, alpha=0.45, label=f'{lbl} N=400' if col_idx==0 else '')
        ax.bar(i + 0.2, v2k,  0.35, color=col, alpha=0.95, label=f'{lbl} N=2000' if col_idx==0 else '')
    ax.set_title(f'Dataset: {ds_names[col_idx]}', fontsize=10)
    ax.set_xticks(range(len(SCALE_MODELS)))
    ax.set_xticklabels([MODEL_LABEL.get(m,m) for m in SCALE_MODELS], rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('F1-macro' if col_idx==0 else '')
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle('Ablação A — Escalabilidade Amostral: N=400 (translúcido) vs N=2000 (sólido)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'fig3_scaling.pdf', dpi=150)
plt.savefig(OUT / 'fig3_scaling.png', dpi=150)
plt.close()
print("fig3 done")

# ── Figure 4: 5 features ablation ────────────────────────────────────────────
sc5f   = collect(noise5, ['TWS_5f','TWM_5f','TWC_5f'])

NOISE_MODELS = ['StandardLSSVM','ADMMNesterovLSSVM','DualFISTA',
                'FTTransformer_sparsemax','FTTransformer_softmax',
                'NystromLSSVMColnorm','FTTransformerCURColnorm']
pairs_5f = [('TWS','TWS_5f'), ('TWM','TWM_5f'), ('TWC','TWC_5f')]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for col_idx, (ds2d, ds5f) in enumerate(pairs_5f):
    ax = axes[col_idx]
    for i, m in enumerate(NOISE_MODELS):
        v2d = np.mean(sc400[m].get(ds2d, [float('nan')])) if sc400[m].get(ds2d) else float('nan')
        v5f = np.mean(sc5f[m].get(ds5f, [float('nan')])) if sc5f[m].get(ds5f) else float('nan')
        col = model_color(m)
        ax.bar(i - 0.2, v2d, 0.35, color=col, alpha=0.45)
        ax.bar(i + 0.2, v5f, 0.35, color=col, alpha=0.95)
    ax.set_title(f'Dataset: {ds_names[col_idx]}', fontsize=10)
    ax.set_xticks(range(len(NOISE_MODELS)))
    ax.set_xticklabels([MODEL_LABEL.get(m,m) for m in NOISE_MODELS], rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel('F1-macro' if col_idx==0 else '')
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle('Ablação B — Robustez a Features de Ruído: 2D (translúcido) vs 5D reajustado (sólido)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'fig4_5features.pdf', dpi=150)
plt.savefig(OUT / 'fig4_5features.png', dpi=150)
plt.close()
print("fig4 done")

# ── Figure 5 (NEW): MK5 results ───────────────────────────────────────────────
scmk = collect(mk5_data, ['MKE','MKM','MKH'])

MK_MODELS = ['StandardLSSVM','ADMMNesterovLSSVM','DualFISTA',
             'FSALSSVm','IPLSSVm','FTTransformer_softmax',
             'NystromLSSVMColnorm','FTTransformerCURColnorm','OppositeMapsLSSVM']
MK_DS = ['MKE','MKM','MKH']
MK_LABELS = ['MKE (fácil)','MKM (médio)','MKH (difícil)']

x = np.arange(len(MK_MODELS))
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for col_idx, ds in enumerate(MK_DS):
    ax = axes[col_idx]
    vals  = [np.mean(scmk[m].get(ds, [float('nan')])) if scmk[m].get(ds) else float('nan')
             for m in MK_MODELS]
    stds  = [np.std(scmk[m].get(ds, [float('nan')])) if scmk[m].get(ds) else 0
             for m in MK_MODELS]
    cols  = [model_color(m) for m in MK_MODELS]
    lbls  = [MODEL_LABEL.get(m,m) for m in MK_MODELS]
    ax.bar(range(len(MK_MODELS)), vals, color=cols, alpha=0.85, yerr=stds, capsize=3)
    ax.set_title(MK_LABELS[col_idx], fontsize=10)
    ax.set_xticks(range(len(MK_MODELS)))
    ax.set_xticklabels(lbls, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel('F1-macro' if col_idx==0 else '')
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle('Ablação C — MK5: Dados Sintéticos com Features Informativas', fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'fig5_mk5.pdf', dpi=150)
plt.savefig(OUT / 'fig5_mk5.png', dpi=150)
plt.close()
print("fig5 done")

print("All figures saved to", OUT)
