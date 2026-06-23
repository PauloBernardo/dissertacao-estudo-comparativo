# ADMMNystromLSSVM — Runbook

Guia para rodar os experimentos e validar se o modelo está sendo útil.

---

## 1. Setup

```bash
git pull origin main
source .venv/bin/activate   # ou: python -m venv .venv && pip install -e .
```

---

## 2. Tier 1 re-run (modelos L1 com grade estendida)

Antes de testar o ADMMNystrom, é preciso re-rodar o Tier 1 para ADMM-N, ADMM-EN,
FISTA e DualFISTA com a grade corrigida (sigma livre + lambda estendido).
Isso vai mostrar a esparsidade real desses modelos para comparação justa.

```bash
SEEDS=$(python -c "print(' '.join(str(i) for i in range(30)))")

python -u scripts/run_tier1_gridcv.py \
    --models ADMMNesterovLSSVM ADMMElasticNet FISTANesterov DualFISTA \
    --datasets AUS BCW GCR HAB PID TWC TWM TWS VCP \
    --seeds $SEEDS \
    --output results/tier1_l1_rerun.json \
    2>&1 | tee logs/tier1_l1_rerun.log
```

**Tempo estimado:** ~8-10h (4 modelos × 9 datasets × 30 seeds, grid=45 combos).
O script é resumível: se cair, basta rodar de novo — ele pula entradas já gravadas.

**O que verificar ao terminar:**

```python
import json
data = json.load(open('results/tier1_l1_rerun.json'))
ok = [r for r in data if r['status'] == 'ok']
print(f'{len(ok)}/1080 completos')

# Esparsidade média por modelo
from collections import defaultdict
sp = defaultdict(list)
for r in ok:
    sp[r['variant']].append(r['sparsity_ratio'])
for m, vals in sp.items():
    print(f"{m:25}  esp={sum(vals)/len(vals):.1%}  n={len(vals)}")
```

**O que esperamos ver:** ADMM-N com >50% esparsidade média (prova HAB mostrou 59%
com lambda=1.0 + sigma=0.5 livres). FISTA e DualFISTA também devem subir.

---

## 3. ADMMNystromLSSVM — Modo A (single-machine, escalável)

### 3a. Smoke test rápido

```bash
python -c "
import numpy as np
from sklearn.datasets import make_classification
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM

X, y = make_classification(n_samples=1000, n_features=15, random_state=0)
y = np.where(y==1, 1.0, -1.0)

m = ADMMNystromLSSVM(sigma=2.0, tau=0.5, lambda_=0.1, m_ratio=0.10,
                     max_iter=500, n_blocks=1, random_state=0)
m.fit(X[:700], y[:700])
acc = np.mean(m.predict(X[700:]) == y[700:])
print(f'acc={acc:.3f}  n_nz={m.n_support_}/{m.m_}  sparsity={m.sparsity_ratio_:.1%}')
"
```

**Esperado:** acc > 0.80, sparsity > 0% (com lambda=0.1 ou 1.0).

### 3b. Rodar no Tier 1 (validação cruzada com datasets conhecidos)

```bash
SEEDS=$(python -c "print(' '.join(str(i) for i in range(30)))")

python -u scripts/run_tier1_gridcv.py \
    --models ADMMNystromLSSVM \
    --datasets AUS BCW GCR HAB PID TWC TWM TWS VCP \
    --seeds $SEEDS \
    --output results/tier1_admm_nystrom.json \
    2>&1 | tee logs/tier1_admm_nystrom.log
```

**Tempo estimado:** ~2-3h (grid=45, mesmo datasets do Tier 1).

**Critério de validação:**

```python
import json, statistics as st
from collections import defaultdict

base = json.load(open('results/tier1_gridcv.json'))      # resultados originais
new  = json.load(open('results/tier1_admm_nystrom.json'))

# F1 do NystromLSSVM original vs ADMMNystrom
nystrom_f1 = [r['test_f1_macro'] for r in base
              if r['variant'] == 'NystromLSSVMColnorm' and r['status'] == 'ok']
admm_nystrom_f1 = [r['test_f1_macro'] for r in new
                   if r['variant'] == 'ADMMNystromLSSVM' and r['status'] == 'ok']

print(f"NystromLSSVM      F1={st.mean(nystrom_f1):.4f}")
print(f"ADMMNystromLSSVM  F1={st.mean(admm_nystrom_f1):.4f}")

# Esparsidade
sp = [r['sparsity_ratio'] for r in new
      if r['variant'] == 'ADMMNystromLSSVM' and r['status'] == 'ok']
print(f"ADMMNystrom sparsity={st.mean(sp):.1%}")
```

**O modelo é útil se:**
- F1 próximo ao NystromLSSVM (perda < 2pp aceitável pelo custo de esparsidade)
- Esparsidade nos landmarks > 20% com lambda=1.0
- Melhor F1 que ADMM-N puro nos datasets grandes (PID, GCR, AUS)

---

## 4. ADMMNystromLSSVM — Modo B (block-parallel, teste do modelo distribuído)

### 4a. Benchmark de speedup

Rode com N grande para que o CᵀC domine o tempo (em N pequeno, o overhead do
joblib supera o ganho):

```bash
python -c "
import numpy as np, time
from sklearn.datasets import make_classification
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM

# Simula Tier 2: N=5000, m=500
N, m_ratio = 5000, 0.10
X, y = make_classification(n_samples=N, n_features=20, random_state=0)
y = np.where(y==1, 1.0, -1.0)

resultados = []
for n_blocks, n_jobs in [(1,1), (2,2), (4,4)]:
    m = ADMMNystromLSSVM(sigma=2.0, tau=0.5, lambda_=0.1, m_ratio=m_ratio,
                         max_iter=200, n_blocks=n_blocks, n_jobs=n_jobs,
                         random_state=0)
    t0 = time.perf_counter()
    m.fit(X, y)
    total = time.perf_counter() - t0
    resultados.append((n_blocks, m.ctc_wall_time_, total))
    print(f'blocks={n_blocks}  ctc={m.ctc_wall_time_:.3f}s  total={total:.2f}s')

base_ctc = resultados[0][1]
print()
for blocks, ctc, _ in resultados:
    print(f'blocks={blocks}  speedup_ctc={base_ctc/ctc:.2f}x')
"
```

**O modelo distribuído é viável se:** speedup CᵀC ≥ 2x com n_blocks=4 para N≥5000.

### 4b. Comparar A vs B no Tier 1 (resultados devem ser idênticos)

```bash
python -u scripts/run_tier1_gridcv.py \
    --models ADMMNystromDistributed \
    --datasets AUS BCW GCR HAB PID \
    --seeds $(python -c "print(' '.join(str(i) for i in range(10)))") \
    --output results/tier1_admm_nystrom_distributed.json

python -c "
import json
a = {r['dataset']+str(r['seed']): r['test_f1_macro']
     for r in json.load(open('results/tier1_admm_nystrom.json'))
     if r['status']=='ok'}
b = {r['dataset']+str(r['seed']): r['test_f1_macro']
     for r in json.load(open('results/tier1_admm_nystrom_distributed.json'))
     if r['status']=='ok'}
common = set(a) & set(b)
diffs = [abs(a[k]-b[k]) for k in common]
print(f'Entradas comuns: {len(common)}')
print(f'Diferença máxima F1 (A vs B): {max(diffs):.2e}')
print('OK' if max(diffs) < 1e-6 else 'DIVERGENCIA — investigar')
"
```

**Esperado:** diferença máxima < 1e-6 (resultados numericamente idênticos).

---

## 5. Resumo dos critérios de validação

| Critério | Aceitável | Ótimo |
|---|---|---|
| F1 ADMMNystrom vs NystromLSSVM | ±2pp | ADMMNystrom ≥ Nystrom |
| Esparsidade landmarks (lambda=1.0) | > 20% | > 50% |
| Speedup CᵀC modo B, N=5000, 4 blocos | ≥ 2x | ≥ 3x |
| A == B (resultados idênticos) | diferença < 1e-6 | exato |
| Tier 1 re-run: esparsidade ADMM-N | > 40% | > 60% |

---

## 6. Mesclar resultados no tier1_gridcv.json (após Tier 1 re-run)

```python
import json

original = json.load(open('results/tier1_gridcv.json'))
rerun    = json.load(open('results/tier1_l1_rerun.json'))

# Remove entradas antigas dos 4 modelos L1
L1 = {'ADMMNesterovLSSVM', 'ADMMElasticNet', 'FISTANesterov', 'DualFISTA'}
original_sem_l1 = [r for r in original if r.get('variant') not in L1]

merged = original_sem_l1 + rerun
merged.sort(key=lambda r: (r.get('variant',''), r.get('dataset',''), r.get('seed',0)))

with open('results/tier1_gridcv.json', 'w') as f:
    json.dump(merged, f)

print(f'Merged: {len(merged)} entradas ({len(original_sem_l1)} mantidas + {len(rerun)} novas)')
```
