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
| StandardLSSVM | `lssvm/standard.py` | suykens1999 | LSSVM/CLASSICOS/Suykens_NeurProcLett.pdf | — | ✅ (2026-09-18) CG Hestenes-Stiefel; b=(1ᵀη)/(yᵀη) ≡ (yᵀμ)/(yᵀη) por simetria de H |
| ADMMNesterovLSSVM | `lssvm/primal/admm_nesterov.py` | marinho2025iwann | IWANN___LSSVM_ADMM.pdf | — | ✅ (converge ao LASSO exato) |
| ADMMElasticNet | `lssvm/primal/admm_nesterov.py` (λ₂>0) | marinho2025iwann | IWANN___LSSVM_ADMM.pdf | — | ✅ (mesmo solver) |
| ADMMNystromLSSVM | `lssvm/primal/admm_nystrom.py` | marinho2025iwann · williams2001nystrom · zhao2022nysadmm | IWANN…; NIPS-2000-nystrom; 2202.11599v2.pdf | arXiv:2202.11599 | ✅ (2026-09-18) base C=K(X,Z), f=Cθ, penalidade nos pesos dos landmarks; ρ=1/λmax(CᵀC/τ); threshold λ/2ρ; esparsidade sobre N |
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
| NystromLSSVMColnorm | `nystrom_lssvm_wrapper.py` | williams2001nystrom · espinoza2006fixed · drineas2005nystrom · kumar2012sampling | NIPS-2000-nystrom; espinoza2006.pdf; drineas05a.pdf | — | ✅ (2026-09-18) Woodbury verificado algebricamente; b=(1ᵀΩ⁻¹y)/(1ᵀΩ⁻¹1); predição via W⁻¹Cᵀα consistente com a extensão de Nyström; colnorm = ‖xᵢ‖² (proxy, como no texto) |
| FTTransformerCURColnorm (FT-CUR) | `ft_transformer_cur_wrapper.py` · `ft_transformer_model.py` | xiong2021nystromformer · mahoney2009cur | Transformers/…/2102.03902v3.pdf; …cur-matrix… | arXiv:2102.03902 | ✅ conceitual (Nyströmformer Alg.1; pinv destacada a testar) |
| SAINTColnorm | `ft_transformer_saint_wrapper.py` · `ft_transformer_model.py::SAINTClassifier` | somepalli2021saint | Transformers/ESTADO DA ARTE/2106.01342_SAINT.pdf | arXiv:2106.01342 | ✅ (2026-09-18, 2ª revisão) fiel ao **código de referência** (`style="reference"`, padrão): pré-norma + residual, FFN GEGLU mult 4, MSA `dim_head=16`, MISA sobre (p+1)·d com `dim_head=64`, FF2 na linha achatada, embedding FC+ReLU(100) por atributo, head MLP(1000) no CLS. `style="paper_eq"` = Eqs. 1–2 do artigo (pós-norma), que **não** coincidem com o código: pós-norma colapsa sob lr 1e-3 (ablação `results/saint_style_ablation.json`). Sem pré-treino contrastivo (declarado). **Resultados publicados = versão só-CLS; o rerun do Kaggle com `paper_eq` também precisa ser refeito com `reference`** |
| FTTransformer (+ atenções esparsas) | `transformers/ft_transformer.py` · `transformers/sparse_attention/*` | gorishniy2021revisiting · martins2016sparsemax · peters2019entmax | Transformers/… | arXiv:2106.11959 | ✅ softmax/top-k/sparsemax (sparsemax = entmax α=2 a 1e-16). ❌→✅ **entmax**: bisseção com intervalo errado (Σp≈1,4–7 antes de renormalizar; corrigido 2026-09-18 com Jacobiana exata) — **resultados FT-Entmax publicados vieram da versão errada** |
| XGBoost | `xgboost_wrapper.py` | chen2016xgboost | — | — | ✅ (biblioteca, n_jobs=1, hist) |

## Rodada de 2026-09-18 — todos os modelos auditados
- **Bug real encontrado:** `sparse_attention/entmax_attention.py` — intervalo de bisseção errado; a saída não era entmax-α (era uma distribuição recortada em z_max − 1/(α−1) e renormalizada). Corrigido com o intervalo de Peters et al. (2019) e backward exato; testes de regressão em `tests/test_transformers.py::TestEntmaxBisectionRegression`. **Pendência:** re-executar FT-Entmax (Tier 1, Tier 2, Ablações A–D, benchmark) ou marcar as linhas publicadas como "variante recortada".
- **Armadilha removida:** `landmark_selection.get_selector('opposite')` era a reflexão OBL de Tizhoosh rotulada como Opposite Maps (origem da confusão no texto). Renomeada para `'obl_reflection'`; `'opposite'` agora levanta erro apontando para `select_opposite_landmarks` (K2M fiel, o único usado nos resultados).
- **SAINT reimplementado fiel** (decisão do usuário: é baseline, tem de ser o modelo do artigo); versão só-CLS preservada como `SAINTClassifierCLSOnly`.
- **Documentado no texto (não é bug):** FT-CUR/SAINT vs FT baselines usam backbones distintos (Adam+clip+MLP head vs AdamW+linear head); esparsidade de atenção conta pesos < 1e-4; ADMM-Nyström é subset-of-regressors (f = K(x,Z)θ), custo O(m²)/iteração; IP usa α com sinal.
- **Ablação A assimétrica (achado 2026-09-18):** `ablation_a_transformers.json` veio de GridSearchCV em N=2000 (`run_tier1_gridcv` nos `*_2k`), enquanto LSSVM/XGBoost transferiram hiperparâmetros do Tier 1; a tese afirma transferência para todos. Corrigido no notebook de re-execução (todos os Transformers via `run_ablation_a_scaling.py --transformers-only`). Ver `docs/rerun_saint_entmax.md`.
- **Orçamento de treino dos Transformers (achado 2026-09-18, ver `scripts/pilot_transformer_budget.py`):** com lotes ≥ N (Tier 1: 200–950 amostras; FT-CUR em lote completo sempre), uma época = um passo de gradiente; o protocolo (40 épocas, paciência 6 sobre val_loss) treina os Transformers com 7–30 passos no Tier 1 e 40 (FT-CUR) a 160 (FT) no Tier 2. Medido: com 400 épocas, FT-Softmax vai de 0,62→0,79 e SAINT de 0,48→0,83 na espiral. Os artigos-fonte usam lr 1e-4, lote 256 e sem teto curto (FT: paciência 16 épocas; SAINT: 100 épocas + melhor ponto de validação). Mesmo mecanismo do artefato de orçamento do ADMM (Ablação D). Decisão de re-execução pendente; `min_epochs` (piso) adicionado a FT, SAINT e FT-CUR.
- **FT-CUR — novos modos (padrão inalterado):** `minibatch_landmarks="global"` (m landmarks globais anexados a cada lote; o modo "per_batch" histórico sorteia landmarks dentro do lote e nunca teve F1 avaliado em dado real) e `predict_mode="streaming"` (resumo só do treino; rota da Tabela 19, agora no wrapper; verificado pareado em N=2000: mesmos rótulos em 5/6 execuções, Δ máx. 0,008).
- **Ressalva de protocolo:** no FT-CUR em mini-batch (só o benchmark N ≥ 10 000), os landmarks de treino são sorteados por batch (`train_epoch`), não colnorm; colnorm vale para o full-batch (Tier 1/2, Ablação D) e para a inferência.


## Método do laudo (por modelo)
1. **Fidelidade algébrica** — comparar equações/algoritmo do paper com o código.
2. **Invariante numérico** — prova independente (converge ao ótimo do LASSO vs `sklearn.Lasso`? degenera no StandardLSSVM no limite sem redução? resíduo KKT ≈ 0?).
3. **Veredito + desvios documentados** + teste de regressão quando aplicável.
