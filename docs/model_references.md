# Mapa Modelo → Paper-fonte + Laudo de Auditoria

Mapeamento de cada modelo implementado para (a) o arquivo de implementação,
(b) a citação no `dissertacao-latex/Referencias.bib`, (c) o PDF do paper-fonte
em `BASE TEORICA/` (mantida **fora** do repositório — papers com copyright, não
versionados), e (d) o veredito da auditoria de fidelidade.

> **Objetivo:** acelerar futuras rodadas de auditoria. Ao reauditar um modelo,
> comece pela linha correspondente: abra o PDF do paper e o arquivo do código.
>
> **Nota de copyright:** os PDFs em `BASE TEORICA/` são papers publicados e
> **não** são commitados (repo público). Para os \*open-access\*, a coluna
> "arXiv/URL" permite recuperá-los.

## Veredito: ✅ fiel · ⚠️ desvia · ❌ não-fiel (deprecado) · ⏳ pendente · 🟢 baixo risco (não auditado)

| Modelo (variante) | Arquivo | Citação (bib) | Paper (BASE TEORICA) | arXiv/URL | Veredito |
|---|---|---|---|---|---|
| StandardLSSVM | `lssvm/standard.py` | suykens1999 | LSSVM/CLASSICOS/Suykens_NeurProcLett.pdf | — | 🟢 |
| ADMMNesterovLSSVM | `lssvm/primal/admm_nesterov.py` | marinho2025iwann | IWANN___LSSVM_ADMM.pdf | — | ✅ (converge ao LASSO exato) |
| ADMMElasticNet | `lssvm/primal/admm_nesterov.py` (λ₂>0) | marinho2025iwann | IWANN___LSSVM_ADMM.pdf | — | ✅ (mesmo solver) |
| ADMMNystromLSSVM | `lssvm/primal/admm_nystrom.py` | marinho2025iwann · williams2001nystrom · zhao2022nysadmm | IWANN…; NIPS-2000-nystrom; 2202.11599v2.pdf | arXiv:2202.11599 | ⏳ (solver ADMM ✅, falta ler o Nyström) |
| FISTANesterov | `lssvm/primal/fista_lssvm.py` | beck2009fista | LSSVM/CLASSICOS/beck2009.pdf | — | ✅ (converge ao LASSO exato) |
| FISTANystrom | `lssvm/primal/fista_nystrom.py` | beck2009fista · williams2001nystrom | beck2009.pdf; NIPS-2000-nystrom | — | ✅ (solver); seleção colnorm = ⏳ |
| DualFISTA | `lssvm/dual/fista_dual_lssvm.py` | beck2009fista · marinho2025iwann | beck2009.pdf; IWANN… | — | ✅ (converge ao ótimo; exploração, não novidade) |
| PCPLSSVm | `lssvm/primal/pcp_lssvm.py` | zhou2016 | LSSVM/CLASSICOS/zhou2016.pdf | — | ✅ (Cholesky pivotada = Alg.1) |
| PruningLSSVM | `lssvm/dual/p_lssvm.py` | suykens2000sparse | LSSVM/CLASSICOS/es2000-352.pdf | — | ✅ (poda por \|α\| = Suykens 2000) |
| **IPLSSVm** (adaptado) | `lssvm/dual/ip_lssvm.py` | carvalho2009 | LSSVM/CLASSICOS/carvalho2009.pdf | — | ❌ QR ≠ critério α → **deprecado** |
| **IPLSSVmOriginal** (fiel) | `lssvm/dual/ip_lssvm_original.py` | carvalho2009 | LSSVM/CLASSICOS/carvalho2009.pdf | — | ✅ critério α-com-sinal + pseudo-inversa |
| **FSALSSVm** (adaptado) | `lssvm/primal/fsa_lssvm.py` | jiao2007fast | LSSVM/CLASSICOS/tnn07a.pdf | — | ⚠️ Matching Pursuit ≠ backfitting → **deprecar** |
| **FSALSSVmOriginal** (fiel) | `lssvm/primal/fsa_lssvm_original.py` | jiao2007fast | LSSVM/CLASSICOS/tnn07a.pdf | — | ✅ backfitting Jiao (Eq.30) |
| **OppositeMapsLSSVM** (adaptado) | `lssvm/dual/opposite_maps.py` | rochaneto2013opposite / neto2013opposite | Opposite Maps…/CLASSICOS/NPL_Ajalmar.pdf | — | ❌ não-fiel (fallback/âncora) → **deprecado** |
| **OppositeMapsOriginalLSSVM** (fiel) | `lssvm/dual/opposite_maps_original.py` | rochaneto2013opposite | …/NPL_Ajalmar.pdf | — | ✅ Kernel k-means + mapa-oposto (passos 3–6) |
| NystromLSSVMColnorm | `nystrom_lssvm_wrapper.py` | williams2001nystrom · espinoza2006fixed · drineas2005nystrom · kumar2012sampling | NIPS-2000-nystrom; espinoza2006.pdf; drineas05a.pdf | — | ⏳ (contribuição — correção interna) |
| FTTransformerCURColnorm (FT-CUR) | `ft_transformer_cur_wrapper.py` · `ft_transformer_model.py` | xiong2021nystromformer · mahoney2009cur | Transformers/…/2102.03902v3.pdf; …cur-matrix… | arXiv:2102.03902 | ✅ conceitual (Nyströmformer Alg.1; pinv destacada a testar) |
| SAINTColnorm | `ft_transformer_saint_wrapper.py` | somepalli2021saint | — | arXiv:2106.01342 | 🟢 |
| FTTransformer (+ atenções esparsas) | `transformers/ft_transformer.py` · `transformers/sparse_attention/*` | gorishniy2021revisiting | Transformers/… | arXiv:2106.11959 | 🟢 |
| XGBoost | `xgboost_wrapper.py` | chen2016xgboost | — | — | 🟢 (biblioteca) |

## Prioridade para a próxima rodada de auditoria
1. 🎯 **NystromLSSVMColnorm** e **FT-CUR** — contribuições (correção interna, não fidelidade a paper)
2. **ADMM-Nyström** — falta ler a parte Nyström (solver ADMM já ✅)
3. 🟢 baixo risco (Standard, XGBoost, FT core, SAINT, atenções esparsas)

## Método do laudo (por modelo)
1. **Fidelidade algébrica** — comparar equações/algoritmo do paper com o código.
2. **Invariante numérico** — prova independente (converge ao ótimo do LASSO vs `sklearn.Lasso`? degenera no StandardLSSVM no limite sem redução? resíduo KKT ≈ 0?).
3. **Veredito + desvios documentados** + teste de regressão quando aplicável.
