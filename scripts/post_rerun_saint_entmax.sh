#!/usr/bin/env bash
# Pós-execução da re-execução SAINT (fiel) + FT-Entmax (corrigido) + Ablação A por transferência.
# Uso:  bash scripts/post_rerun_saint_entmax.sh [--no-merge] [--skip-surfaces] [--metadataset CSV]
#   --no-merge       pula a etapa 1 (útil para regenerar/validar a partir dos JSONs canônicos atuais)
#   --skip-surfaces  não regenera as superfícies de decisão (treina Transformers; lento em CPU)
#   --metadataset    caminho do metadataset_clean.csv do TabZilla (senão pula a validação externa)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
THESIS=../dissertacao-latex
V="SAINTColnorm FTTransformer_entmax"; TAG=saintentmax
MERGE=1; SURF=1; META=""
while [ $# -gt 0 ]; do case "$1" in
  --no-merge) MERGE=0;; --skip-surfaces) SURF=0;; --metadataset) META="$2"; shift;; *) echo "arg desconhecido: $1"; exit 1;; esac; shift; done

step(){ echo; echo "──────── $* ────────"; }

if [ $MERGE = 1 ]; then
  step "1. merge dos JSONs de re-execução"
  for f in tier1 tier2 n5000 ablA ablBC table19; do [ -f results/rerun_saint_entmax_$f.json ] || { echo "faltando results/rerun_saint_entmax_$f.json"; exit 1; }; done
  $PY scripts/merge_rerun_results.py --target results/tier1_gridcv.json --source results/rerun_saint_entmax_tier1.json --variants $V --tag $TAG
  $PY scripts/merge_rerun_results.py --target results/tier2_transformers.json --source results/rerun_saint_entmax_tier2.json --variants $V --tag $TAG
  $PY scripts/merge_rerun_results.py --target results/tier2_fixedparams_n5000_transformers.json --source results/rerun_saint_entmax_n5000.json --variants $V --tag $TAG
  $PY - <<'PYX'
import json
r=json.load(open('results/rerun_saint_entmax_ablBC.json')); B={'TWS_5f','TWM_5f','TWC_5f'}
json.dump([x for x in r if x['dataset'] in B], open('results/rerun_saint_entmax_ablB.json','w'))
json.dump([x for x in r if x['dataset'] not in B], open('results/rerun_saint_entmax_ablC.json','w'))
PYX
  $PY scripts/merge_rerun_results.py --target results/ablation_b_noise.json --source results/rerun_saint_entmax_ablB.json --variants $V --tag $TAG
  $PY scripts/merge_rerun_results.py --target results/ablation_c_mk5.json   --source results/rerun_saint_entmax_ablC.json --variants $V --tag $TAG
  cp results/ablation_a_transformers.json results/ablation_a_transformers_pre_${TAG}_backup.json
  cp results/rerun_saint_entmax_ablA.json results/ablation_a_transformers.json
  echo "ablation_a_transformers.json substituído inteiro (seis Transformers por transferência)"
  $PY scripts/merge_rerun_results.py --target results/table19_results.json --source results/rerun_saint_entmax_table19.json --variants SAINT_minibatch SAINT_fullbatch FTTransformer_entmax --tag $TAG
  # dumps antigos do Tier 2 (lidos por generate_tier2_n5000_tables / extract_tier2_fixed_params): fora do caminho
  mkdir -p results/_dumps_antigos && for f in "results/tier2_transformers (1).json" "results/tier2_transformers (2).json" results/tier2_transformers_merged.json results/tier2_transformers_pre2_backup.json; do [ -f "$f" ] && git mv -k "$f" results/_dumps_antigos/ 2>/dev/null || { [ -f "$f" ] && mv "$f" results/_dumps_antigos/; }; done; true
fi

step "2. combinar Tier 2 e gerar tabelas de resultados"
$PY scripts/combine_tier2_results.py
$PY scripts/generate_analysis.py --results results/tier1_gridcv.json --prefix tier1
$PY scripts/generate_analysis.py --results results/tier2_combined.json --prefix tier2
$PY scripts/generate_analysis.py --results results/tier2_transformers.json --prefix tier2_transformers

step "3. ablações, N=5000, métricas, Tabela 19, figuras"
$PY scripts/generate_ablation_tables.py
$PY scripts/generate_ablation_figs.py
$PY scripts/generate_report_figs.py
$PY scripts/generate_tier2_n5000_tables.py
$PY scripts/generate_metrics_tables.py
$PY scripts/make_table19.py --input results/table19_results.json --output $THESIS/tables/benchmark_transformers_memory.tex --n-values 1000,5000,10000,20000,50000
$PY scratch/regen_figures/plot_scaling.py || echo "(plot_scaling falhou — figuras do benchmark não regeneradas)"
if [ $SURF = 1 ]; then $PY scripts/plot_decision_surfaces.py; else echo "(superfícies de decisão puladas)"; fi
if [ -n "$META" ]; then $PY scripts/compare_tabzilla.py --metadataset "$META"; else echo "(TabZilla pulado: passe --metadataset)"; fi

step "4. copiar para a tese ([ht]→[H]) e normalizar decimais"
for t in tier1_results tier1_ranks tier1_sparsity tier1_wilcoxon_raw tier2_results tier2_ranks tier2_sparsity tier2_wilcoxon_raw ablation_a ablation_b ablation_c tier2_n5000_comparison tier2_n5000_ftcur_m; do
  sed 's/\[ht\]/[H]/' results/tables/$t.tex > $THESIS/tables/$t.tex; done
for f in fig1_f1_all_models fig2_sparsity_tradeoff fig3_scaling fig4_5features fig5_mk5 fig6_decision_surfaces_lssvm_n400 fig7_decision_surfaces_transformers_n400 fig8_decision_surfaces_lssvm_n2000 fig9_decision_surfaces_transformers_n2000; do
  [ -f results/report_figs/$f.pdf ] && cp results/report_figs/$f.pdf $THESIS/Figuras/$f.pdf; done
$PY scripts/generate_nemenyi_analysis.py     # lê os ranks com ponto decimal: ANTES de normalizar
$PY scripts/normalize_decimals.py --apply >/dev/null

step "5. reaplicar edições manuais (linhas dos Transformers em tier2_sparsity; resizebox nas ablações)"
$PY - <<'PYX'
import re, pathlib
T = pathlib.Path('../dissertacao-latex/tables')
# (a) tier2_sparsity: acrescenta as linhas dos Transformers (de tier2_transformers_sparsity), se ausentes
sp = (T/'tier2_sparsity.tex').read_text(); tr = pathlib.Path('results/tables/tier2_transformers_sparsity.tex').read_text()
rows = [l for l in tr.splitlines() if re.match(r'^(FT-|SAINT)', l.strip())]
rows = [re.sub(r'(\d)\.(\d)', r'\1{,}\2', l.strip()) for l in rows]
have = set(l.split('&')[0].strip() for l in sp.splitlines() if '&' in l)
new = [r for r in rows if r.split('&')[0].strip() not in have]
if new:
    body = [l for l in sp.splitlines()]
    i = next(k for k,l in enumerate(body) if '\\bottomrule' in l)
    data = [l for l in body[:i] if '&' in l and 'Model' not in l and 'Sparsity' not in l] + new
    data.sort(key=lambda l: l.split('&')[0].strip().lower())
    head = body[:next(k for k,l in enumerate(body) if '\\midrule' in l)+1]
    sp = '\n'.join(head + data + body[i:]) + '\n'
    (T/'tier2_sparsity.tex').write_text(sp); print(f'tier2_sparsity: +{len(new)} linhas de Transformers')
# (b) resizebox nas tabelas largas
for t in ['ablation_a','ablation_b']:
    p = T/f'{t}.tex'; s = p.read_text()
    if '\\resizebox' not in s:
        s = s.replace('  \\begin{tabular}{lcccccccccc}', '  \\resizebox{\\textwidth}{!}{%\n  \\begin{tabular}{lcccccccccc}').replace('  \\end{tabular}\n\\end{table}', '  \\end{tabular}}\n\\end{table}')
        p.write_text(s); print(f'{t}: resizebox reaplicado')
PYX
# (c) legendas curtas com espaço não separável ("Tier~1"), como nas versões publicadas
sed -i 's/--- Tier \([12]\)\]/--- Tier~\1]/' $THESIS/tables/tier1_metrics.tex $THESIS/tables/tier2_metrics.tex $THESIS/tables/tier1_nemenyi.tex $THESIS/tables/tier2_nemenyi.tex

step "6. recompilar a dissertação"
( cd $THESIS && pdflatex -interaction=nonstopmode -halt-on-error Dissertacao.tex >/dev/null && bibtex Dissertacao >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error Dissertacao.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error Dissertacao.tex >/dev/null && echo "PDF ok: $(pdfinfo Dissertacao.pdf | grep Pages)" && (grep -aE "Overfull|undefined|multiply" Dissertacao.log | grep -v microtype || echo "log limpo") )
echo; echo "Pronto. Agora revisar o TEXTO: números de SAINT, FT-Entmax e Ablação A (ver docs/rerun_saint_entmax.md, passo 6)."
