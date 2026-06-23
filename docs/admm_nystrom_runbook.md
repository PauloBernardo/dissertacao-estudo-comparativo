# ADMMNystromLSSVM — Runbook

Como rodar e validar o modelo novo nas outra máquina.

---

## Setup

```bash
git pull origin main
source .venv/bin/activate
```

---

## Modo A — single-machine (validação no Tier 1)

Roda o ADMMNystromLSSVM nos 9 datasets do Tier 1 para comparar com o
NystromLSSVM existente e ver se a esparsidade por L1 vale a pena.

```bash
SEEDS=$(python -c "print(' '.join(str(i) for i in range(30)))")

python -u scripts/run_tier1_gridcv.py \
    --models ADMMNystromLSSVM \
    --datasets AUS BCW GCR HAB PID TWC TWM TWS VCP \
    --seeds $SEEDS \
    --output results/tier1_admm_nystrom.json \
    2>&1 | tee logs/tier1_admm_nystrom.log
```

**Tempo estimado:** ~2-3h (grid=45 combos, 9 datasets × 30 seeds).

**Validar ao terminar:**

```python
import json, statistics as st

base = json.load(open('results/tier1_gridcv.json'))
new  = json.load(open('results/tier1_admm_nystrom.json'))

nystrom_f1 = [r['test_f1_macro'] for r in base
              if r['variant'] == 'NystromLSSVMColnorm' and r['status'] == 'ok']
admm_f1    = [r['test_f1_macro'] for r in new
              if r['status'] == 'ok']
sp         = [r['sparsity_ratio'] for r in new if r['status'] == 'ok']

print(f"NystromLSSVM      F1={st.mean(nystrom_f1):.4f}")
print(f"ADMMNystromLSSVM  F1={st.mean(admm_f1):.4f}  sparsity={st.mean(sp):.1%}")
```

**É útil se:** F1 próximo ao NystromLSSVM (perda < 2pp) com sparsidade > 20%.

---

## Modo B — block-parallel (teste do modelo distribuído)

Valida que os blocos paralelos dão o mesmo resultado e medem speedup real.

### Verificar que A == B

```bash
python -u scripts/run_tier1_gridcv.py \
    --models ADMMNystromDistributed \
    --datasets AUS BCW GCR HAB PID \
    --seeds $(python -c "print(' '.join(str(i) for i in range(10)))") \
    --output results/tier1_admm_nystrom_dist.json
```

```python
import json

a = {r['dataset']+str(r['seed']): r['test_f1_macro']
     for r in json.load(open('results/tier1_admm_nystrom.json'))
     if r['status'] == 'ok'}
b = {r['dataset']+str(r['seed']): r['test_f1_macro']
     for r in json.load(open('results/tier1_admm_nystrom_dist.json'))
     if r['status'] == 'ok'}

comum = set(a) & set(b)
diffs = [abs(a[k] - b[k]) for k in comum]
print(f'Diferença máxima F1 A vs B: {max(diffs):.2e}')
# Esperado: < 1e-6
```

### Benchmark de speedup (precisa N grande)

```python
import numpy as np, time
from sklearn.datasets import make_classification
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM

X, y = make_classification(n_samples=5000, n_features=20, random_state=0)
y = np.where(y == 1, 1.0, -1.0)

base_ctc = None
for n_blocks, n_jobs in [(1,1), (2,2), (4,4)]:
    m = ADMMNystromLSSVM(sigma=2.0, tau=0.5, lambda_=0.1, m_ratio=0.10,
                         max_iter=200, n_blocks=n_blocks, n_jobs=n_jobs,
                         random_state=0)
    m.fit(X, y)
    if base_ctc is None:
        base_ctc = m.ctc_wall_time_
    print(f'blocks={n_blocks}  ctc={m.ctc_wall_time_:.3f}s  '
          f'speedup={base_ctc/m.ctc_wall_time_:.2f}x')
```

**Esperado com N=5000:** speedup ≥ 2x com 4 blocos.
Com N < 1000 o overhead do joblib domina — resultado irrelevante.

---

## Critérios resumidos

| Teste | Passou |
|---|---|
| F1 ADMMNystrom vs NystromLSSVM | perda < 2pp |
| Sparsidade landmarks (lambda=1.0) | > 20% |
| A == B (resultados idênticos) | diferença < 1e-6 |
| Speedup CᵀC com 4 blocos, N=5000 | ≥ 2x |
