# STATUS — Dissertação Tier 2 (atualizado 30/05/2026, 22:00)

Resumo do estado atual dos experimentos, achados confirmados, hipóteses abertas e próximos passos.

---

## 🏃 Experimentos rodando agora

### Script A — Tuning ADMM (CPU)

- **PID**: `1031578`
- **Comando**: `python scripts/run_tier2_n5000.py --seeds 30 --trials 20 --folds 3 --models-group cpu_all --params-file results/tuning/best_params_tier2_n5000_cpu.json --output-file results/tier2_n5000_cpu.json`
- **Log**: `/tmp/tier2_cpu.log`
- **Estado**: tunando `ADMMNesterovLSSVM__HIGGS50K` (iniciado 17:39 do dia 30/05)
- **Restante**: ADMM-N HIGGS50K + ADMM-EN × 6 datasets + experimentos finais
- **ETA**: **01/06 manhã/tarde**

### Script C — Experimentos paralelos (CPU)

- **PID**: `1524265`
- **Comando**: `python scripts/run_parallel_tuned.py --seeds 30`
- **Log**: `/tmp/tier2_parallel.log`
- **Estado**: rodando `DualFISTA` (130/180), depois `SAINT` (0/180)
- **Restante**: ~230 runs
- **ETA**: **31/05 madrugada/manhã**

### Como checar os 2 ao mesmo tempo

```bash
echo "=== A ===" && tail -3 /tmp/tier2_cpu.log
echo "=== C ===" && tail -3 /tmp/tier2_parallel.log
ps -p 1031578,1524265 -o pid,etime,pcpu,pmem
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader
```

### Como ver progresso por modelo do Script C

```bash
source .venv/bin/activate && python3 - <<'EOF'
import json
from collections import defaultdict
data = json.load(open('results/tier2_n5000_parallel.json'))
by = defaultdict(int)
for r in data:
    if r.get('status') == 'ok':
        by[r.get('model_variant') or r.get('model')] += 1
for m in sorted(by, key=lambda x: -by[x]):
    print(f"  {m:<28} {by[m]:>3}/180")
EOF
```

---

## ✅ Achados confirmados

### 1. Nyström-SVM é o melhor trade-off esparsidade × F1 (Tier 1)

- F1 = 0.834 com 69% de esparsidade
- Apenas 0.008 abaixo do LSSVM-Std denso
- Auditoria matemática completa em `scripts/audit_nystrom_lssvm.py`

### 2. DualFISTA empata com LSSVM-Std (Tier 1)

- F1 = 0.842 com 20% de esparsidade
- Melhor método em baixa esparsidade
- **Hipótese de eficiência** (a confirmar): 10-100× mais rápido que ADMM-Nesterov (paper-base)

### 3. SAINT ≈ FT-CUR no Tier 1 (compressão Nyströmformer é fiel)

- SAINT (denso) 0.744 vs FT-CUR (79% compressão) 0.747
- Diferença trivial: comprimir não custa F1 no Tier 1

### 4. ⭐ H2 confirmada: `val_loss` >> `val_acc` em early stopping (NOVO)

Ablação H2 (`scripts/run_early_stop_ablation.py`, dados em `results/tier2_early_stop_ablation.json`):

| Modelo | Dataset | val_acc | val_loss | Δ | Wilcoxon p |
|--------|---------|--------:|---------:|---:|-----------:|
| FT-CUR | CREDIT | 0.438 | 0.657 | +0.219 | 1.3×10⁻⁶ |
| FT-CUR | BANK | 0.507 | 0.714 | +0.207 | 1.9×10⁻⁹ |
| SAINT | CREDIT | 0.439 | 0.684 | +0.245 | 9.3×10⁻¹⁰ |
| SAINT | BANK | 0.473 | 0.710 | +0.237 | 9.3×10⁻¹⁰ |

**O colapso aparente em CREDIT/BANK era um artefato metodológico, não limitação estrutural.**

---

## 🔬 Hipóteses abertas

| H | Causa | Status | Como testar |
|---|-------|--------|-------------|
| H1 | Imbalanceamento + colnorm não-estratificado | **A rodar depois** (`run_tier2_balanced.py --group cpu_all`) | Tier 2 com `--balance-train` |
| H2 | `val_acc` em early stopping favorece classe majoritária | **CONFIRMADA** ✅ (p < 10⁻⁵) | Ablação concluída |
| H3 | Limitação fundamental do CUR | Enfraquecida pelo H2; pequeno resíduo em CREDIT (FT-CUR) | Por exclusão |

---

## 📊 Arquivos de resultados

### Tier 1 + Ablações (concluído)

- `results/tier1_results.json` — 16 modelos baseline × 9 datasets × 30 seeds
- `results/tier1_custom_models.json` — Nyström, FT-CUR, SAINT, ADMM-Variantes
- `results/synthetic_scaling_n2000.json` — Ablação A (escala)
- `results/synthetic_5features.json` / `_tuned.json` — Ablação B (ruído)
- `results/synthetic_mk5.json` — Ablação C (MK5)

### Tier 2 N=5000 (em andamento)

- `results/tier2_n5000_cpu.json` — Script A (sendo escrito)
- `results/tier2_n5000_gpu.json` — Script B (✅ concluído ontem)
- `results/tier2_n5000_parallel.json` — Script C (sendo escrito)
- **Merge final**: precisa ser feito quando tudo terminar (`tier2_n5000_*.json` → consolidado)

### Tier 2 N=5000 — ablação H2 (concluído)

- `results/tier2_early_stop_ablation.json` — 360 runs (FT-CUR/SAINT × CREDIT/BANK × 3 metrics)
- `results/tuning/best_params_early_stop_ablation.json` — params tunados por métrica

### Snapshots de tuning para reuso/análise

- `results/tuning/best_params_tier2_n5000_snapshot.json` — versão congelada
- `results/tuning/best_params_tier2_n5000_cpu.json` — sendo escrito por Script A
- `results/tuning/best_params_tier2_n5000_gpu.json` — concluído

---

## 📋 Próximos experimentos planejados

### 1. Re-rodar FT-CUR/SAINT Tier 2 com `val_loss` (CRÍTICO)

Os resultados atuais de FT-CUR/SAINT no Tier 2 usaram `val_acc` (bias confirmado). Precisamos revalidar com `val_loss`.

**Onde rodar**: Colab T4 GPU (preparado, ~6-8h)
**Notebook**: https://colab.research.google.com/github/PauloBernardo/dissertacao-estudo-comparativo/blob/main/notebooks/ftcur_saint_valloss_colab.ipynb
**Comando local equivalente**:
```bash
python scripts/run_ftcur_saint_rerun.py \
    --early-stop-metric val_loss \
    --datasets ADULT CREDIT BANK TELCO SHOPPERS HIGGS50K \
    --output-file results/tier2_n5000_ftcur_saint_valloss.json \
    --params-file results/tuning/best_params_ftcur_saint_valloss.json \
    --seeds 30 --trials 20 --folds 3
```

### 2. H1: Tier 2 balanceado (todos os 18 modelos)

```bash
# CPU group (LSSVMs + SAINT) — ~30h
python scripts/run_tier2_balanced.py --group cpu_all

# GPU group (4 FT baselines + FT-CUR) — ~6h
python scripts/run_tier2_balanced.py --group gpu_transformer
```

Importante: para FT-CUR/SAINT no H1, usar protocolo corrigido (`val_loss`):
```bash
python scripts/run_ftcur_saint_rerun.py \
    --early-stop-metric val_loss \
    --datasets CREDIT BANK SHOPPERS \
    --balance-train \
    --output-file results/tier2_balanced_ftcur_saint.json \
    --params-file results/tuning/best_params_tier2_balanced_ftcur_saint.json
```

### 3. Tier 2 full-data (Colab)

- `notebooks/tier2_colab_cpu.ipynb` — XGBoost + Nyström-SVM em N completo (~3-5h)
- `notebooks/tier2_colab_gpu.ipynb` — 4 FT + FT-CUR em N completo (~10-12h)

---

## 🎯 Tarefas pendentes pós-experimentos

Quando todos os experimentos terminarem:

1. **Merge dos arquivos JSON do Tier 2**:
   - `tier2_n5000_cpu.json` + `tier2_n5000_gpu.json` + `tier2_n5000_parallel.json` → `tier2_n5000_final.json`
   - Substituir FT-CUR/SAINT pelos resultados `val_loss` (manter `val_acc` como histórico)

2. **Atualizar `relatorio.tex`**:
   - Tabela final do Tier 2 N=5000 (18 modelos × 6 datasets) com val_loss para FT-CUR/SAINT
   - Tabela H1 (balanceado vs imbalanceado)
   - Tabela Tier 2 full-data
   - Substituir RASCUNHO da seção 10.5 com dados consolidados

3. **Benchmark de eficiência (item 5 das Próximas etapas)**:
   - Extrair `train_time_s` de todos os JSONs
   - Plotar DualFISTA vs ADMM-N vs ADMM-EN vs FISTA-N: tempo médio por fit
   - Validar hipótese "ADMM-EN converge mais rápido que ADMM-N por Tikhonov"

4. **Regenerar figuras**:
   ```bash
   python scripts/generate_report_figs.py
   ```

5. **Recompilar PDF**:
   ```bash
   cd results && pdflatex relatorio.tex && pdflatex relatorio.tex
   ```

6. **Commit + push final**

---

## 🌳 Estado do git

- **Branch**: `main`
- **Último commit**: `49b2601` "Add future work: computational efficiency benchmark DualFISTA vs ADMM"
- **Remote**: https://github.com/PauloBernardo/dissertacao-estudo-comparativo
- **Sync**: local = origin

### Arquivos não commitados (em uso pelos scripts)

```
results/tier2_n5000_cpu.json       # sendo escrito por Script A
results/tier2_n5000_parallel.json  # sendo escrito por Script C
results/tuning/best_params_tier2_n5000_cpu.json  # vivo
results/tuning/best_params_tier2_n5000_gpu.json  # concluído mas não commitado
results/tuning/best_params_tier2_n5000.json      # legado
results/tuning/_combined_t1_tmp.json             # temp
```

Não commitar enquanto experimento roda (race condition na leitura).

---

## 📁 Scripts importantes

| Script | Função |
|--------|--------|
| `scripts/run_tier2_n5000.py` | Experimento principal Tier 2 (suporta `--no-cap`, `--balance-train`, `--models-group`) |
| `scripts/run_parallel_tuned.py` | Experimentos paralelos com params já tunados (Script C atual) |
| `scripts/run_tier2_balanced.py` | Wrapper H1: chama `run_tier2_n5000.py` com `--balance-train` |
| `scripts/run_early_stop_ablation.py` | Ablação H2 (val_acc/val_loss/val_f1_macro) |
| `scripts/run_ftcur_saint_rerun.py` | Rerun focado FT-CUR/SAINT com metric configurável |
| `scripts/audit_nystrom_lssvm.py` | Auditoria matemática do Nyström-SVM (já passou) |
| `scripts/generate_report_figs.py` | Gera figuras do relatório |

---

## 🔧 Decisões metodológicas importantes

1. **N_TRAIN=5000** no Tier 2 (subsample N=7143, split 70/30)
2. **`val_loss`** é o critério de early stopping correto para FT-CUR/SAINT (não `val_acc`)
3. **Balanceamento apenas no treino** (`balance_train=True` no runner), preservando teste original
4. **SAINT em CPU localmente** (MX350 2GB não cabe atenção n×n); GPU OK no Colab T4
5. **`use_inter_instance`** detecta corretamente se modelo precisa de contexto na validação
6. **CREDIT_BAL/BANK_BAL/SHOPPERS_BAL** são loaders deprecated (emitem warning); usar `--balance-train` no runner

---

## 🌡️ Monitoramento de hardware

Comando rápido pra ver temp + processos:
```bash
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader
sensors 2>/dev/null | grep "Package id" | head -1
ps -p 1031578,1524265 -o pid,etime,pcpu,pmem
```

Limites:
- GPU MX350 thermal limit: ~95°C (atual ~71°C, ok)
- CPU thermal limit: 100°C (atual ~75°C, ok)

---

## 📞 Quem é quem

- **paulobernardo06@gmail.com** — autor da dissertação (mestrando)
- **Orientador** — Felipe P. Marinho (autor do LSSVM-ADMM-Nesterov original)
- **Próxima entrega**: dissertação completa com Tier 2 consolidado, ablações H1/H2 e benchmark de eficiência
