# Plano de fechamento da dissertação

**Decisão estruturante (2026-09-18, do autor):** o protocolo principal **permanece GridSearchCV por
semente + 30 sementes**. O trunfo da dissertação não é ganhar alguns pontos de F1, é ser um estudo
robusto e reproduzível: com busca por semente, cada semente tuna e avalia de forma independente, e é
isso que sustenta o desenho pareado do Friedman/Nemenyi. Optuna tunado uma vez por dataset acoplaria as
30 sementes a uma única busca estocástica — mais barato e com F1 possivelmente maior, mas menos
reprodutível e com menos poder estatístico. Portanto: **nada do que já foi rodado é descartado**; o
achado do orçamento de treino entra como análise (Seção 5.7) e o protocolo alternativo entra como
**apêndice complementar**, não como substituto.

Última atualização: 2026-09-18 (sessão de auditoria). Este documento é o ponto de partida de cada
sessão até a defesa. Um passo de cada vez; marcar `[x]` ao concluir e anotar a data.

## 0. Onde as coisas estão
| O quê | Onde |
|---|---|
| Tese (LaTeX) | `../dissertacao-latex/` — `Dissertacao.tex`, capítulos `Introducao/Capitulo1–4/Conclusao/Apendice1.tex`, `tables/`, `Figuras/` |
| Código do estudo (git, branch `revisao/estatistica-e-proveniencia`) | este repositório; a branch padrão `main` está DESATUALIZADA |
| Mapa modelo → paper → laudo | `docs/model_references.md` |
| Pipeline pós-re-execução (merge → regerar → copiar → normalizar → Nemenyi → edições manuais → compilar) | `scripts/post_rerun_saint_entmax.sh` (validado: reproduz as tabelas publicadas a partir dos JSONs canônicos) e `docs/rerun_saint_entmax.md` |
| Notebook Kaggle da re-execução (SAINT fiel + FT-Entmax + Ablação A por transferência) | `notebooks/saint_entmax_rerun_kaggle.ipynb` (faz checkout da branch certa; progresso enxuto por fase; saídas `rerun_saint_entmax_{tier1,tier2,n5000,ablA,ablBC,table19}.json`) |
| Merge de resultados | `scripts/merge_rerun_results.py --target … --source … --variants … --tag …` (backup automático) |
| Piloto do protocolo dos artigos | `scripts/pilot_transformer_budget.py` |
| Tira-teima BANK N=5000 (CPU) | `results/tiebreak_bank_n5000_{fixed,armE,gridcv}.json` |
| Papers-fonte (fora do git) | `../BASE TEORICA/` (SAINT e FT-Transformer salvos em `Transformers/ESTADO DA ARTE/`) |

## 1. O que a auditoria de 2026-09-18 estabeleceu
1. **Todos os 20 modelos auditados** (laudos em `docs/model_references.md`). Bug real: **entmax** não era entmax (intervalo de bisseção errado) → corrigido, com testes. **SAINT** era uma simplificação (inter-instâncias só no CLS) → reimplementado fiel a Somepalli et al. 2021 (`SAINTClassifier`; antigo = `SAINTClassifierCLSOnly`). `get_selector('opposite')` era reflexão OBL, não Opposite Maps → renomeado `obl_reflection`.
2. **Regressão de dados corrigida:** `results/tier2_transformers.json` estava com registros pré-Newton-Schulz do FT-CUR (sobrescrito em 13/07). Restaurado; tabelas regeradas; texto ajustado (XGBoost vence os 6, FT-CUR 0,698 / rank 6,833, Δ Ablação D +0,0048, Transformers +0,0218, Spearman Tier 2 0,827/0,970/0,714).
3. **Ablação A era assimétrica:** Transformers re-tunados por GridCV em N=2000; LSSVM/XGBoost transferidos. A tese diz "transferidos". O notebook re-executa os SEIS Transformers por transferência (`run_ablation_a_scaling.py --transformers-only`).
4. **Orçamento de treino dos Transformers (achado central):** com lote ≥ N, 1 época = 1 passo de gradiente. Protocolo (40 épocas, paciência 6 sobre val_loss em 21–43 pontos de validação) → 7–30 passos no Tier 1; no Tier 2, 160 (FT), 80 (SAINT), 40 (FT-CUR lote completo). Com 400 épocas, espiral: FT-Softmax 0,62→0,79, SAINT 0,48→0,83. É o MESMO mecanismo do artefato de orçamento do ADMM (Ablação D). Artigos: FT-Transformer = AdamW lr 1e-4, lote 256, paciência 16 épocas, sem teto; SAINT = AdamW lr 1e-4, lote 256, 100 épocas, melhor ponto de validação.
5. **FT-CUR:** rota de predição dos Tiers (contexto treino+teste) ≠ rota da Tabela 19 (streaming, só treino). Verificado pareado em N=2000: mesmos rótulos em 5/6 execuções, Δ máx. 0,008 → números publicados válidos. Modo mini-lote histórico (landmarks sorteados por lote) nunca teve F1 avaliado em dado real. Novos modos no wrapper, padrão inalterado: `minibatch_landmarks="global"`, `predict_mode="streaming"`, `min_epochs`.
6. **Tira-teima BANK N=5000 (CPU, protocolo publicado; CONCLUÍDO 19:50):** FT-CUR fixo 0,636 ± 0,093 (2/10 colapsos) vs re-tunado 0,659 ± 0,046 (0/3 colapsos; sementes 0,595/0,685/0,697) → a queda da Ablação D é colapso sob transferência, revertida por re-tuning. SAINT fiel fixo 0,699 ± 0,027 (0/10) e re-tunado 0,694 ± 0,030 (0,736/0,680/0,667) — re-tuning não muda o SAINT. Diferença re-tunada SAINT − FT-CUR = +0,035 com faixas sobrepostas (3 sementes): SAINT à frente, mas não decisivo. Parada por val_f1_macro piora (5/10 colapsos). Publicado em N=2000: FT-CUR 0,644 vs SAINT antigo 0,592. JSONs: `results/tiebreak_bank_n5000_{fixed,armE,gridcv}.json`.
7. **SAINT — 2ª revisão (a queda brutal era o estilo de normalização).** As Eqs. 1–2 do artigo descrevem
   **pós**-norma (`LN(f(x)) + x`); o código que gerou os resultados do artigo (`somepago/saint`,
   `RowColTransformer`, estilo `colrow`) usa **pré**-norma (`x + f(LN(x))`), FFN **GEGLU** (mult 4),
   `dim_head=64` na atenção de linha e **FF2 sobre a linha achatada** $(n\,d)$, não por token.
   Sob o protocolo publicado (lr 1e-3, sem aquecimento) a pós-norma colapsa
   (`results/saint_style_ablation.json`, 10 sementes, configuração fixa):

   | dataset | pós-norma (Eqs. 1–2) | pré-norma (código) | publicado (só-CLS) |
   |---|---|---|---|
   | TWS | 0,480 (5/10 colapsos) | **0,624 (0/10)** | 0,641 |
   | TWC | 0,335 (10/10) | **0,438 (6/10)** | 0,460 |
   | AI4I | 0,819 (0/10) | **0,858 (0/10)** | 0,808 |
   | VCP | 0,703 (2/10) | **0,771 (1/10)** | 0,815 |
   | HAB | 0,540 (3/10) | 0,504 (5/10) | 0,534 |

   Com `style="reference"` (agora o padrão) o SAINT fiel volta ao patamar dos números publicados, em vez
   de cair 0,067 no Tier 1. A pré-norma é também o que os FT do estudo usam, então a comparação passa a
   ser entre iguais nesse aspecto. **Consequência: a fase do SAINT no rerun do Kaggle (feita com
   `paper_eq`) precisa ser refeita; as fases do FT-Entmax permanecem válidas** (modelo independente).
8. **Piloto do protocolo dos artigos lançado** (19:53, CPU, 3 sementes, TWS/HAB/AI4I/BANK/TELCO): saída incremental em `results/pilot_transformer_budget.json`, log em scratchpad da sessão. Ver §3.

## 2. VERSÃO 1 — fechar com implementações fiéis, protocolo publicado, limitação declarada
- [x] **1.0a** [CONCLUÍDO 19/09 — entmax rodado nas 5 fases restantes em duas T4; Tier 1 já vinha da sessão anterior.] **Kaggle: FT-Entmax** — notebook `notebooks/entmax_rerun_2gpu_kaggle.ipynb` (duas T4, sem
  Tier 1, sem SAINT). Saídas `results/entmax_{tier2,n5000,ablA,ablBC,table19}.json`. O Tier 1 do entmax já
  está mesclado em `results/tier1_gridcv.json` (300 registros; F1 0,7424; zeros 0,0798) e os 9 parquets
  derivados foram versionados para eliminar escrita concorrente em `data/raw/`.
  *(Histórico do item original:)* **Kaggle, em curso: só FT-Entmax.** O notebook `saint_entmax_rerun_kaggle.ipynb` foi
  alterado (commit `ab7e79d`) para rodar apenas `FTTransformer_entmax`; a Ablação A roda os **cinco**
  Transformers sem o SAINT e a Tabela 19 só a variante do entmax. Tier 1 do entmax já está em
  `~/Downloads/rerun_saint_entmax_tier1.json` (600 registros, inclui SAINT `paper_eq` que **não** se usa).
  Os registros de SAINT dos JSONs desta execução são descartados.
- [x] **1.0b** [CONCLUÍDO 19/09 — SAINT `style="reference"` rodado nas 6 fases (300+180+180+60+180+14 registros, zero erro).] **SAINT com `style="reference"`** — notebook pronto:
  `notebooks/saint_reference_2gpu_kaggle.ipynb` (duas T4; Tier 1, Tier 2, Ablação D, acréscimo do SAINT à
  Ablação A já rodada, Ablações B/C e as duas linhas da Tabela 19 em uma placa). Saídas
  `results/saint_{tier1,tier2,n5000,ablA,ablBC,table19}.json`. A célula 2 verifica o estilo no clone.
  Custo medido: **≈9 h na T4** ou **≈25 h na MX350 local** (medido com o runner oficial e o estilo novo:
  Tier 1/GCR 104 s por semente, Tier 2/BANK 270 s por semente, 31 ajustes cada).
  **A Tabela 19 tem de ficar na T4**: o SAINT em lote completo estoura os 2 GB da MX350 em N=10 000.
  Se o resto rodar local, declarar em nota de rodapé que o SAINT foi executado em outra GPU.
- [x] **1.1** [CONCLUÍDO 19/09 — todos os JSONs baixados e validados (completude conferida por fase).] Kaggle: concluir as 6 fases do notebook; baixar os JSONs para `results/`.
- [x] **1.2a** **FT-Entmax mesclado e tabelas regeneradas (2026-09-19).** Todas as fases (Tier 1, Tier 2,
  Ablação D, Ablações A/B/C, Tabela 19) mescladas nos JSONs canônicos com backups `*_pre_entmax_backup.json`;
  dumps antigos do Tier 2 movidos para `results/_dumps_antigos/`. Pipeline rodado, PDF com 133 páginas, log limpo.
  **O que mudou:** Tier 2 do entmax 0,7010 → 0,7090 (zeros 0,064 → 0,205); ele passa a ser o **2º melhor
  \textit{rank} do Tier 2** (3,33, atrás só do XGBoost) e supera o \textit{softmax} nos 6 datasets
  ($p = 0{,}011$ sobre 180 pares), embora bem abaixo da CD de 9,4. Tier 1 inalterado (0,7424; empate com o
  \textit{softmax}, $p = 0{,}75$). Tabela 19: treino 2,5× o \textit{softmax} (era "várias vezes").
  Ablação A: os cinco Transformers mudaram (agora por transferência) — FT-Sparsemax lidera (0,774), FT-CUR
  último (0,716).
  **Texto já ajustado:** nova subseção §Dose-resposta da esparsidade de atenção (`sec:esparsidade_tier2_dose`),
  bullets do Tier 1 e do Tier 2, esparsidade do Tier 2, eficiência, Discussão, Conclusão item 2, Resumo e Abstract.
- [x] **1.2b** [CONCLUÍDO 19/09 — merge do SAINT e Ablação A reescrita.] Após o SAINT: refazer o merge só do SAINT, regenerar e **reescrever a Ablação A inteira**
  (item 1.4) — o parágrafo atual cita os Δ antigos (FT-Softmax +0,190, FT-Sparsemax +0,262, SAINT +0,014),
  todos substituídos pelos novos (+0,106, +0,123, e o do SAINT a sair).
- [x] **1.2** [CONCLUÍDO 19/09 — pipeline rodado duas vezes (entmax e SAINT); PDF 133 p., log limpo.] `bash scripts/post_rerun_saint_entmax.sh` (faz merge, regenera, copia, normaliza, Nemenyi, reaplica edições manuais, compila). Conferir o resumo de diffs que ele imprime.
- [x] **1.3** [CONCLUÍDO 19/09 — Tier 1 bullets, Tier 2 bullets, esparsidade, métricas (Spearman), Ablação D, eficiência, Conclusão itens 1/2/5, Resumo e Abstract.] Texto — números de SAINT e FT-Entmax: Cap. Resultados (Tier 1 bullets e Friedman; §esparsidade Transformers; Tier 2 bullets, esparsidade e métricas; Ablação D §SAINT; benchmark inter-instâncias), Conclusão (itens 1, 2, 5, 6), Apêndice (Nemenyi, métricas complementares), Resumo/Abstract só se alguma conclusão mudar.
- [x] **1.4** [CONCLUÍDO 19/09 — dois parágrafos do platô da espiral substituídos; o platô era artefato da versão só-CLS.] Texto — Ablação A inteira dos Transformers (agora por transferência; os seis mudam). Reescrever o parágrafo da espiral/SAINT e a decomposição por semente com os dados novos.
- [ ] **1.5** Texto — limitação do orçamento de treino: (a) Metodologia §Ambiente/Transformers: uma frase com os passos efetivos por tier e a referência aos regimes dos artigos; (b) Limitações: item novo; (c) Trabalhos futuros: protocolo em passos alinhado aos artigos (= versão 2).
- [ ] **1.6** Apêndice curto "Ablação de orçamento de treino": tabela espiral/HAB × {protocolo, 40 ép. lote 32, 400 ép.} para FT-Softmax e SAINT (dados desta sessão; ideal repetir em GPU com 10 sementes nos 3 sintéticos e nos 6 modelos — ~1 h de GPU).
- [ ] **1.7** Texto — Ablação D: parágrafo de validação por re-tuning do FT-CUR no BANK (item 1.6 acima do §1), espelhando o do ADMM-Nyström; qualificar "SAINT é o que mais se beneficia de N" (depende do BANK; sobrevive ao re-tuning nesse dataset).
- [ ] **1.8** Texto — suavizar: "data-hungry" → "sob orçamento fixo de 40 épocas"; "FT-CUR empata com SAINT" → "sob o mesmo orçamento"; verbos do Resumo/Conclusão em linha com o poder do Nemenyi (CD ≈ 9,4).
- [ ] **1.9** Itens do juízo de valor ainda abertos: parágrafo "o que fica de meu" (Intro + Conclusão); limitação do orçamento de tuning assimétrico (LSSVM 36–75 configs vs Transformers 6); título ("formulações duais" no plural); Folha de Aprovação (membros); parágrafos longos da Discussão; apêndice de reprodutibilidade (commit + mapa JSON→tabela + script).
- [ ] **1.10** Nota no Cap. 5: variantes de mini-lote da Tabela 19 são configurações de custo, sem F1 avaliado; rota streaming verificada equivalente (Δ ≤ 0,008).
- [ ] **1.12** **Seção 5.7 — unificar o orçamento de otimização nas duas famílias.**
  *Insumos já medidos (19/09):* (a) passos de gradiente por época por regime e modelo (Tier 1: 1 para todos;
  Tier 2: 4 FT / 2 SAINT / 1 FT-CUR; N=5000: 8 / 4 / 1); (b) ablação de orçamento na espiral e no HAB
  (FT-Softmax 0,645→0,961 e SAINT 0,547→0,922 com 400 épocas); (c) épocas em que a paciência 6 corta
  (FT-Softmax 7–19, SAINT 9–30); (d) taxa de colapso do SAINT × dificuldade do \textit{dataset}
  (Spearman $-0{,}63$, $p = 0{,}009$; não correlaciona com $p$ nem com $N$); (e) FT-CUR: mais passos **não**
  ampliam o ganho com $N$ (`results/ftcur_batch_confound.json`) → gargalo é a fidelidade do resumo;
  (f) SAINT em mini-lote ganha $+0{,}097$ ao dobrar $N$ (`results/saint_batch_context.json`), mas o braço
  de lote completo em $N=5000$ **não rodou** (VRAM de 2 GB na MX350) — logo, no lado do SAINT, passos e
  contexto ficam \emph{confundidos}. Escrever sem atribuir o mecanismo ao SAINT, ou rodar os 15 ajustes
  faltantes numa T4 antes. A seção hoje é
  "O Orçamento de Iterações como Explicação da Ablação D" (só ADMM). Renomear para algo como
  "O Orçamento de Otimização: Iterações do ADMM e Épocas dos Transformers" e acrescentar uma subseção com:
  (a) o mecanismo — com lote ≥ N, uma época = um passo de gradiente; o protocolo dá 7–30 passos no Tier 1
  e 40 (FT-CUR em lote completo) a 160 (FT) no Tier 2, com a paciência 6 cortando dentro do platô
  (medido: FT-Softmax para na época 7–19, SAINT na 9–30, em validação de 21–43 pontos);
  (b) a evidência — ablação de orçamento (espiral: FT-Softmax 0,645→0,961 e SAINT 0,547→0,922 com 400
  épocas; HAB muda pouco) e a taxa de colapso do SAINT fiel no Tier 1 concentrada nos sintéticos
  geométricos e no HAB, com as medianas das sementes que treinam competitivas;
  (c) o paralelo explícito com o ADMM — mesmo mecanismo (orçamento fixo em iterações que não é invariante
  à escala do problema), mesmos dois caminhos (mais orçamento × outro ponto que converge no orçamento),
  e a mesma conclusão de que o caminho robusto é dar orçamento;
  (d) a ressalva de leitura — as comparações *dentro* da família Transformer permanecem válidas (mesmo
  orçamento para os seis), a comparação com LSSVM/XGBoost (resolvidos ao ótimo) é que fica enviesada.
- [ ] **1.13** **Novo apêndice (5º) — "Protocolo de treino alternativo para os Transformers".** Apêndice
  complementar, explicitamente fora do protocolo principal: descreve o protocolo ancorado nos artigos
  (FT-Transformer: AdamW, Optuna/TPE com orçamento em iterações, *lr* LogUniform[1e-5,1e-3], paciência 16,
  sem teto; SAINT: 100 épocas, melhor ponto de validação), apresenta o piloto (§3.1) e o estudo F3 (§3.3)
  como *probe* de robustez, e conclui se as conclusões do corpo mudam. Deixar claro por que NÃO substitui
  o protocolo principal (item acima: reprodutibilidade e desenho pareado).
- [ ] **1.11** Recompilar; conferir log (0 overfull, 0 undefined); memória do projeto atualizada; commit + push (dados, tabelas, docs).

## 3. ESTUDO COMPLEMENTAR (apêndice) — protocolo de treino ancorado nos artigos
**Não substitui o corpo.** Entra como o 5º apêndice (item 1.13): mostra o que acontece quando os
Transformers recebem o orçamento de treino dos artigos, medindo se as conclusões do estudo principal
se sustentam. O protocolo principal (GridSearchCV por semente, 30 sementes) permanece intocado.
Protocolo proposto (`PROTOCOLS["paper"]` em `scripts/pilot_transformer_budget.py`):
AdamW, lr 1e-4, sem agenda; lote 256; parada após 16 épocas sem melhora na validação, **piso de 200 épocas** (única regra nossa: nos datasets em que 1 época = 1 passo), teto 1000; melhor ponto de validação; decaimento de peso por artigo (1e-5 FT, 0,01 SAINT); arquiteturas e grades como estão. FT-CUR: decidir entre lote completo (desenho original) e mini-lote com landmarks globais (`minibatch_landmarks="global"` + `predict_mode="streaming"`).
- [x] **2.1** Piloto — CONCLUÍDO 2026-09-18 20:47 (ver §3.1/3.2 abaixo).
- [ ] **2.1-bis** Piloto na CPU: `python scripts/pilot_transformer_budget.py --datasets TWS HAB AI4I BANK TELCO --seeds 3` (≈3–4 h). Responde: o piso tira os modelos do platô? lr 1e-4 basta? FT-CUR mini-lote global se sustenta em F1? custo por ajuste → horas de GPU.
- [ ] **2.2** Decidir alcance: (a) seis Transformers em tudo (Tier 1, Tier 2, Ablações A–D, Tabela 19): estimativa 120–200 h de GPU (publicado consumiu ≈40 h: 10,6 Tier 1 + 17,8 Tier 2 + 5,9 Abl. A + 4,6 Abl. B/C + 1,2 N=5000); (b) só SAINT e FT-CUR nos Tiers + ablação de orçamento nos sintéticos para os seis: ≈30–40 h.
- [ ] **2.3** Implementar o perfil de protocolo em `src/tuning/grids.py` (novo conjunto de `fixed`), notebook Kaggle por fases (copiar o esquema do atual), rodar.
- [ ] **2.4** Merge em JSONs NOVOS (jamais sobrescrever os da versão 1); tabelas próprias, prefixo `apx_protocolo_`; comparar conclusão por conclusão com o corpo.
- [ ] **2.5** Escrever o apêndice (item 1.13) com esses números; no corpo, apenas a referência cruzada a partir da Seção 5.7.

## 4. Regras que já custaram caro (não esquecer)
- Chavear registros por `variant`, não por `model`. Fontes canônicas: `tier1_gridcv.json`, `tier2_gridcv.json` + `tier2_transformers.json`, `tier2_fixedparams_n5000_*.json`, `ablation_{a_scaling,a_transformers,b_noise,c_mk5}.json`, `table19_results.json`.
- Dumps antigos `tier2_transformers (1)/(2)/_merged/_pre2_backup.json` são lidos por `extract_tier2_fixed_params.py` (ordem de glob põe "(1)" primeiro → moda antiga). O script pós-execução move-os para `results/_dumps_antigos/`.
- Tabelas com edição manual que os geradores não reproduzem: linhas dos Transformers em `tier2_sparsity.tex`; legenda longa de `benchmark_lssvm.tex`; `resizebox` em `ablation_a/b.tex`; `Tier~1` nas legendas curtas. O script reaplica todas exceto a legenda do `benchmark_lssvm.tex` (não é regerada).
- `generate_nemenyi_analysis.py` lê os ranks com ponto decimal → rodar ANTES de `normalize_decimals.py`.
- O notebook Kaggle clona `main` por padrão: sempre fazer checkout da branch de trabalho.

### 3.1 Resultado do piloto (2026-09-18, 3 sementes; TWS/HAB na CPU, AI4I/BANK/TELCO na GPU MX350; `results/pilot_transformer_budget.json`)
F1-macro (épocas efetivas, segundos por ajuste):

| dataset | modelo | publicado | artigos lr 1e-4 | artigos lr 1e-3 |
|---|---|---|---|---|
| TWS | FT_softmax | 0.645 (9 ép., 1 s) | 0.805 (299 ép., 4 s) | 0.961 (222 ép., 3 s) |
| TWS | SAINT | 0.547 (22 ép., 0 s) | 0.655 (200 ép., 2 s) | 0.922 (207 ép., 3 s) |
| TWS | FTCUR_full | 0.642 (13 ép., 0 s) | 0.626 (211 ép., 2 s) | 0.774 (235 ép., 2 s) |
| TWS | FTCUR_mb_global | 0.642 (13 ép., 0 s) | 0.621 (208 ép., 2 s) | 0.760 (214 ép., 2 s) |
| HAB | FT_softmax | 0.511 (9 ép., 0 s) | 0.446 (200 ép., 2 s) | 0.594 (200 ép., 2 s) |
| HAB | SAINT | 0.514 (29 ép., 0 s) | 0.591 (227 ép., 3 s) | 0.621 (200 ép., 2 s) |
| HAB | FTCUR_full | 0.608 (16 ép., 0 s) | 0.569 (209 ép., 2 s) | 0.608 (200 ép., 2 s) |
| HAB | FTCUR_mb_global | 0.608 (16 ép., 0 s) | 0.569 (209 ép., 2 s) | 0.608 (200 ép., 2 s) |
| AI4I | FT_softmax | 0.771 (14 ép., 1 s) | 0.887 (200 ép., 10 s) | 0.882 (200 ép., 12 s) |
| AI4I | SAINT | 0.829 (37 ép., 1 s) | 0.850 (205 ép., 11 s) | 0.875 (203 ép., 12 s) |
| AI4I | FTCUR_full | 0.823 (36 ép., 1 s) | 0.835 (425 ép., 12 s) | 0.881 (203 ép., 6 s) |
| AI4I | FTCUR_mb_global | 0.823 (36 ép., 1 s) | 0.846 (313 ép., 25 s) | 0.888 (200 ép., 16 s) |
| BANK | FT_softmax | 0.710 (28 ép., 3 s) | 0.688 (200 ép., 28 s) | 0.669 (200 ép., 30 s) |
| BANK | SAINT | 0.672 (28 ép., 2 s) | 0.677 (204 ép., 31 s) | 0.680 (201 ép., 30 s) |
| BANK | FTCUR_full | 0.602 (40 ép., 3 s) | 0.684 (392 ép., 26 s) | 0.685 (200 ép., 13 s) |
| BANK | FTCUR_mb_global | 0.602 (40 ép., 3 s) | 0.682 (200 ép., 53 s) | 0.703 (200 ép., 49 s) |
| TELCO | FT_softmax | 0.690 (18 ép., 2 s) | 0.708 (200 ép., 35 s) | 0.686 (202 ép., 32 s) |
| TELCO | SAINT | 0.702 (28 ép., 3 s) | 0.709 (200 ép., 32 s) | 0.704 (200 ép., 33 s) |
| TELCO | FTCUR_full | 0.716 (40 ép., 3 s) | 0.701 (280 ép., 21 s) | 0.712 (200 ép., 15 s) |
| TELCO | FTCUR_mb_global | 0.716 (40 ép., 3 s) | 0.708 (200 ép., 53 s) | 0.691 (200 ép., 55 s) |

**Leituras.** (a) Tier 1 é orçamento, não viés indutivo: espiral FT-Softmax 0,645→0,961, SAINT 0,547→0,922 (LSSVM 0,994). (b) lr 1e-4 dos artigos exige piso muito maior que 200 épocas nas bases pequenas; lr 1e-3 com a paciência dos artigos é o que sai do platô. (c) No regime N=2000, FT-Softmax e SAINT quase não mudam (já tinham 80–160 passos); quem ganha é o FT-CUR em lote completo (BANK 0,602→0,685), que tinha 40 passos. (d) FT-CUR mini-lote com landmarks globais iguala ou supera o lote completo (BANK 0,703) — candidato a modo principal, com custo O(N·m) genuíno. (e) FT-Softmax no BANK cai um pouco com mais épocas (0,710→0,669): paciência 16 + piso 200 deixa sobreajustar; calibrar teto/paciência.

### 3.2 Conta de GPU (T4) para a versão 2
Fator de custo por ajuste (protocolo dos artigos lr 1e-3 ÷ publicado, N=2000): FT ×11,5; SAINT ×11,6; FT-CUR lote completo ×4,9; FT-CUR mini-lote global ×18,4. Nos datasets pequenos (1 passo/época) ≈ ×20.
Publicado (T4): Tier 1 10,6 h; Tier 2 17,8 h; Abl. A 5,9 h; Abl. B/C 4,6 h; N=5000 1,2 h; total ≈ 40 h.

As horas abaixo são em **T4 (Kaggle)**; a MX350 local é ≈3× mais lenta (T4 ≈ 0,9–1,05 s por ajuste em N=2000 contra 2,7–2,9 s na MX350; `fit_time_s` dos JSONs publicados é o GridSearchCV inteiro, 31 ou 61 ajustes).

| Opção | O que muda | T4 (Kaggle, 30 h/semana) | MX350 local (contínua) |
|---|---|---|---|
| A | tudo como a v1: grade + 5 folds + 30 sementes, seis modelos | ≈ 560 h → 19 semanas | ≈ 1600 h — inviável |
| B | idem com 10 sementes | ≈ 190 h → 6–7 semanas | ≈ 570 h |
| C | **configuração fixa por modelo (padrão dos artigos / moda da v1), sem grade**, 30 sementes, seis modelos | ≈ 15–20 h → 1 semana | ≈ 45–60 h → 2–3 dias |
| D | só SAINT e FT-CUR com grade + 5 folds, 30 sementes | ≈ 125–190 h → 4–6 semanas | ≈ 400–570 h |
| E | C + grade só no par SAINT/FT-CUR no Tier 2 | ≈ 60–80 h → 2–3 semanas | ≈ 200 h |

### 3.3 Optuna — a opção que segue o artigo e resolve o desvio de \emph{lr}
O FT-Transformer **não** usa grade: usa Optuna (TPE), orçamento em **iterações** (100 no espaço A, 50 no B),
tuna **uma vez por dataset** sobre a partição de validação e depois roda **15 sementes** com a configuração
vencedora (Seção 5.2 e Apêndice E.4 do artigo). O espaço (Tabela 13) inclui camadas [1,4], dimensão de
\emph{embedding} [64,512], três \emph{dropouts}, fator da FFN, **\emph{lr} LogUniform[1e-5, 1e-3]** e
\emph{weight decay} LogUniform[1e-6, 1e-3].

Três consequências para a versão 2:
1. **O desvio de \emph{lr} deixa de existir.** Em vez de nós escolhermos 1e-3 (contra o 1e-4 "padrão"),
   o \emph{lr} entra no espaço de busca do próprio artigo, cujo topo é exatamente 1e-3 — o valor que o
   piloto mostra ser o que sai do platô nas bases pequenas. Nenhuma regra nossa sobra no protocolo.
2. **Resolve a assimetria de \emph{tuning}** apontada na revisão: hoje os Transformers têm 6–12
   configurações (só blocos × cabeças) contra 36–75 dos LSSVMs. Com 30–50 \emph{trials} sobre um espaço
   que inclui \emph{lr}, \emph{dropout}, profundidade e dimensão, passam a ter busca comparável ou mais rica.
3. **O custo é viável**, porque tunar uma vez por (modelo, dataset) troca `30 × 61 ajustes` por
   `n_trials + 30`. O repositório já tem `src/tuning/bayesian.py` (Optuna 4.8 instalado).

Custo dos SEIS Transformers em Tier 1 + Tier 2 + Abl. B/C (Abl. A é transferência; N=5000 é config fixa, ≈1 h):

| Opção | Ajustes por (modelo, dataset) | T4 | MX350 | Kaggle 30 h/sem |
|---|---|---|---|---|
| A — grade × 5 folds × 30 sementes (como a v1) | 930 | 338 h | 1013 h | 11,3 sem |
| C — configuração fixa, 30 sementes | 30 | 10 h | 31 h | 0,3 sem |
| **F1 — Optuna 30 trials (1 split) + 30 sementes** | 60 | 21 h | 63 h | 0,7 sem |
| F2 — Optuna 50 trials (1 split) + 30 sementes | 80 | 28 h | 83 h | 0,9 sem |
| **F3 — Optuna 30 trials × 3 splits + 30 sementes** | 120 | 42 h | 125 h | 1,4 sem |
| F4 — Optuna 50 trials × 3 splits + 30 sementes | 180 | 63 h | 188 h | 2,1 sem |

**Escolhido para o apêndice: F3** (≈42 h de T4 ou ≈125 h na MX350 local, ≈5 dias contínuos).
Como é apêndice, pode rodar depois de a versão 1 estar fechada e compilada, sem bloquear a defesa.
O objetivo de cada \emph{trial} é o F1-macro médio de validação em 3 \emph{splits},
o que evita que a configuração fique colada à partição da semente 0 (no artigo o problema não existe,
porque há uma única partição por dataset; aqui cada semente re-particiona). Cabe em ≈5 dias na MX350
local ou ≈1,5 semana de cota do Kaggle. F1 é o plano B se o tempo apertar.

Espaço de busca proposto (adaptado da Tabela 13 aos tamanhos deste estudo): camadas UniformInt[1,4];
dimensão do \emph{token} {32, 64, 128, 192}; cabeças {2, 4, 8}; \emph{attention/FFN dropout} Uniform[0, 0,5];
\emph{lr} LogUniform[1e-5, 1e-3]; \emph{weight decay} LogUniform[1e-6, 1e-3]; para o FT-CUR, \emph{m\_ratio}
{0,05; 0,1; 0,2} e modo de lote fixo em mini-lote 256 com \emph{landmarks} globais. Orçamento de treino:
lote 256, teto 300 épocas, paciência 16 sobre a validação com piso de 100 épocas, melhor ponto de validação.

Antiga recomendação (mantida como registro): **C** (é o desenho do artigo do FT-Transformer: "default configuration performs on par with tuned"), declarando que na v2 os Transformers usam configuração fixa enquanto os LSSVMs mantêm a grade — assimetria oposta à atual (hoje a grade dos Transformers tem 6–12 configurações contra 36–75 dos LSSVMs) e menos grave, pois o que a v2 quer medir é o efeito do orçamento de treino. Abl. A já é por transferência (barata). Decisão pendente.

---

## Ablação de orçamento de otimização — desenho e achados de 2026-09-19

Substitui o desenho esboçado nos itens 1.6 e 1.12 (que previa congelar hiperparâmetros e variar
épocas). Aquele desenho **não é válido**, pelo motivo levantado pelo orientando: a seleção da grade
acontece *sob* o orçamento apertado, então favorece o que treina rápido, e congelar a escolha
subestimaria o efeito do orçamento.

### Evidência do viés de seleção (dados já existentes)

- SAINT escolhe `n_layers=1` em **53%** (Tier 1) e **63%** (Tier 2), contra 33% do uniforme — a
  configuração mais rasa, a mais rápida de treinar.
- Nas quatro variantes FT a seleção é **quase uniforme** (16–20% nas seis células; uniforme = 16,7%):
  em 40 épocas o escore de CV não distingue arquitetura. A moda que `extract_tier2_fixed_params.py`
  calcula é, nesses casos, moda de ruído.
- top-k e sparsemax puxam no sentido oposto (4 blocos/4 cabeças em 28%/21%), então o viés não tem
  direção única — mais uma razão para deixar a grade re-selecionar dentro de cada braço.
- Confirmação direta: no teste local HAB/seed0, ao trocar 40/6 por 200/16 a arquitetura escolhida
  mudou (FT-softmax 3 blocos/4 cabeças → 2/2; SAINT 2 cabeças → 4).
- Corolário inesperado: reproduzindo HAB/seed0 em outra placa, FT-softmax bate exatamente (0,6510)
  mas o SAINT cai para 0,4250 contra 0,5671 publicado, **porque a grade escolheu outra arquitetura**.
  Com o escore de CV empatado, ruído numérico inverte o argmax. Não é divergência de versão; é
  sintoma do mesmo achado.

### Correção do enquadramento: quem para o treino é a paciência, não o teto

Os `n_epochs` do piloto sob o protocolo publicado mostram que o teto de 40 quase nunca é alcançado nas
bases pequenas: FT-softmax para no TWS em 10/9/8 e no HAB em 10/7/11; FT-CUR no TWS em 18/10/10.
Com lote 512 ≥ conjunto de ajuste, **uma época é um passo de gradiente**, então o critério efetivo é
"pare após 6 passos sem melhora de `val_loss`". Todo texto que disser "teto de 40 épocas apertou"
está errado — o correto é "a paciência de 6 cortou em 7–30 passos".

### A assimetria que sustenta a Seção 5.7 (já com dado em mão)

| | LSSVM (ADMM) | Transformers |
|---|---|---|
| regra de parada | tolerância nos resíduos primal/dual, `tol=1e-6` | paciência sobre `val_loss` |
| o que ela certifica | proximidade do ótimo do problema posto | que parou de melhorar em amostra retida |
| teto | 500 iterações | 40 épocas |
| teto aperta? | **não** — braço "teto 500" em `results/admm_stability_knobs.json`: CREDIT trunca em 500 contra 673–812 do livre e o F1 é idêntico até a 4ª casa; HIGGS50K converge em ~80 | a medir |

### Instrumentação adicionada

`best_epoch` (época do checkpoint restaurado), `n_steps`, `steps_per_epoch` e `stopped_early` passaram
a ser gravados nos dois laços de treino (`ft_transformer_model.py` para SAINT/FT-CUR;
`transformers/ft_transformer.py` tem laço próprio) e expostos pelos wrappers. Os runners de Tier 1 e
Tier 2 ganharam `--budget-epochs/--budget-patience/--budget-min-epochs`, aplicados aos `fixed` de cada
variante respeitando as duas convenções de chave (`max_epochs` no FT, `epochs` no SAINT/FT-CUR), e
gravam as métricas no registro junto de `budget_override`.

`min_epochs` fica DESLIGADO: forçar piso destruiria a medição de onde a paciência dispara — foi o que
inviabilizou o piloto de 18/09 para fins de custo (usava `min_epochs=200`).

### Execução: `notebooks/budget_ablation_2gpu_kaggle.ipynb`

Braço novo = `--budget-epochs 200 --budget-patience 16` (paciência dos artigos), com a grade
re-selecionada dentro do braço. O braço de 40/6 **não** é reexecutado: `results/tier1_gridcv.json` e
`results/tier2_transformers.json` já são esse braço; o pareamento usa as sementes 0–9.

1. **Calibração** (células 1–5, ≈1h30): 2 sementes, Tier 1 completo + Tier 2 em BANK/HIGGS50K/TELCO,
   nos dois orçamentos. Mede a razão de custo real e serve de controle de reprodutibilidade.
2. **Leitura** (célula 6): razão de custo, `best_epoch`/`n_epochs`/passos por modelo e braço, quantos
   bateram no teto, e extrapolação da execução completa.
3. **Completo** (células 7–8): 10 sementes, Tier 1 (10 datasets) e Tier 2 (6 datasets).

Sanidade local (MX350, 2026-09-19): razão de custo 1,7× no HAB (não 5×, porque a paciência ainda
dispara antes do teto). No Tier 2/TELCO, FT-softmax com 200/16 parou na época 24 com `best_epoch=8`
— indício de que no Tier 2 o orçamento **não** aperta, coerente com o piloto (em N=2000 mais épocas
não ajudavam e no BANK pioravam). Se a calibração confirmar, a Seção 5.7 fica: orçamento aperta no
Tier 1 (1 passo/época) e não aperta no Tier 2, o que delimita a conclusão em vez de ampliá-la.

---

## FT-CUR e SAINT não eram determinísticos (achado de 2026-09-19)

Verificação pedida pelo orientando ("parece que tá vivendo 2 tipos dele aqui"). A suspeita estava
certa, mas a causa principal não era bifurcação de modelo: era **semeadura incompleta**.

### O bug

`FTTransformerCURColnorm.fit` e `SAINTColnorm.fit` semeavam só o numpy
(`np.random.RandomState(random_state)`), que governa a seleção de landmarks e o split de validação, e
**nunca semeavam o torch**, que governa a inicialização de pesos e o dropout. O `FTTransformer` das
variantes FT já chamava `torch.manual_seed` (`transformers/ft_transformer.py:281`); os dois wrappers de
atenção inter-instâncias, não.

Consequência medida (AI4I, semente 0, três execuções idênticas): FT-CUR devolvia F1 0,8859 / 0,8953 /
0,8859 com `best_epoch` 73 / 149 / 89; SAINT devolvia 0,9107 / 0,8924 / 0,8579. O FT-softmax repetia
0,8932 nas três.

**Magnitude.** Ruído de execução com semente FIXA, como fração do desvio padrão entre as 30 sementes
publicadas: HAB 85% (FT-CUR) e 78% (SAINT); AI4I 55% e 49%. Ou seja, de metade a quase todo o "desvio
entre sementes" desses dois modelos era ruído de inicialização, não variância de partição — e as barras
de erro deles não são comparáveis às dos outros 18 modelos.

**Corrigido** com `torch.manual_seed` + `torch.cuda.manual_seed_all` nos dois `fit`. Verificado:
determinístico com a mesma semente, e ainda variando entre sementes.

**Não enviesa as médias** (o ruído é simétrico), então as conclusões publicadas seguem válidas; o que
está errado é a *atribuição* da incerteza e a afirmação de reprodutibilidade. Consequências a escrever:
(a) o apêndice de reprodutibilidade (item 1.9) não pode afirmar que a mesma semente reproduz os números
de FT-CUR e SAINT publicados; (b) a tabela de estabilidade precisa de uma nota; (c) no Nemenyi o ruído
extra nos ranks desses dois é conservador, não otimista.

Também corrige um diagnóstico meu anterior: a divergência do SAINT em HAB/seed0 (0,5671 publicado
contra 0,4250 local) foi atribuída a "ruído numérico entre placas". Não era placa; era esta linha.

### Os quatro interruptores do FT-CUR, agora medidos sem o ruído

Com a semeadura corrigida, 4 datasets × 3 sementes, arquitetura fixa
(`results/probe_ftcur_switches_local.json`):

| interruptor | idêntico ao braço do estudo | veredito |
|---|---|---|
| `predict_mode="streaming"` | 12/12 | não-operação; Δ = 0 exato — confirma a equivalência do item 1.10 |
| `minibatch_landmarks="global"` | 12/12 | não-operação com lote completo, como a docstring diz |
| `pinv_grad=True` | 2/12 | **bifurcação real**: ΔF1 +0,0056 em média (−0,017 a +0,048) |

`pinv_grad=True` é a pseudo-inversa Newton-Schulz **diferenciável**, isto é, o Nyströmformer original;
o estudo roda com ela destacada (`no_grad`). Continua sendo o item pendente de auditoria do FT-CUR, e
agora está quantificado: efeito pequeno e de sinal misto, logo é questão de declarar a escolha, não de
refazer o estudo.

Os outros dois modos de atenção do módulo (`cur_full`, que materializa a matriz n×n em O(n²), e
`linear_cur`) são **inalcançáveis pelo estudo**: o único construtor de `FTTransformerClassifier` é o
wrapper do FT-CUR, que fixa `attn_mode="nystrom"`, e nenhum script ou grade passa `attn_mode`. O
`cur_full` só é o default do `nn.Module`, e `SAINTBlock`/`SAINTClassifierCLSOnly` existem apenas para
proveniência da versão só-CLS, sem uso em script algum.

### Armadilha do n×n fechada, e a conta do retrabalho

O padrão de `attn_mode` em `FTTransformerClassifier` era `"cur_full"`, que materializa
A ∈ ℝ^{n×n} e custa O(n²d) — o que **anula a razão de existir do FT-CUR**, cujo ganho é justamente não
pagar O(n²). Nenhum resultado publicado foi afetado (o wrapper sempre passou `"nystrom"`
explicitamente, e é o único construtor no repositório), mas era armadilha para código futuro. Em
2026-09-19 o padrão passou a ser `"nystrom"` e o *fallback* silencioso do dispatcher virou `ValueError`.

`cur_full` fica como **referência apenas**: a CUR ali incide sobre a matriz de atenção verdadeira,
enquanto `"nystrom"` normaliza C, R e W separadamente, então comparar os dois mede o erro de
aproximação do caminho honesto — em n pequeno, e nunca como modelo do estudo.

### Campanha de reexecução (consequência da semeadura)

Contando só os arquivos que alimentam tabela ou figura, e apenas as variantes afetadas
(`SAINTColnorm`, `FTTransformerCURColnorm` e os três seletores `FTTransformerCUR{Random,Kmeans,Opposite}`):

| alvo | h de ajuste | em 2 T4 |
|---|---|---|
| Tier 1 (`tier1_gridcv.json`) | 3,30 | |
| Tier 2 (`tier2_transformers.json`) | 5,78 | |
| Ablação D (`tier2_fixedparams_n5000_transformers.json`) | 0,34 | |
| Ablações A, B, C | 1,94 | |
| Tabela 19 | ~0 | |
| apêndice de seleção de landmarks (`ftcur_sel_tier1.json`, `ftcur_scarce_m10_geo.json`) | 10,02 | |
| **subtotal, protocolo publicado 40/6** | **21,4** | **10,7 h** |
| braço de orçamento 200/16, 30 sementes (Tier 1 + Tier 2) | 42,6 | 21,3 h |
| **campanha completa** | **64,0** | **32 h ≈ 3 sessões** |

Ganho colateral: com os dois braços semeados corretamente, o pareamento do experimento de orçamento
fica limpo e dispensa a declaração de assimetria entre arma publicada e arma nova.

Risco a vigiar na releitura: a seleção de grade passa a ser determinística, e como o escore de CV quase
não separa as arquiteturas (16–20% nas seis células), as arquiteturas escolhidas VÃO mudar. As médias
não devem se mover muito (o ruído era simétrico), mas números citados no texto precisam ser reconferidos
um por um — em especial a posição do FT-CUR entre os Transformers no Tier 1 e a média do SAINT.

---

## O que o seletor `colnorm` realmente mede (2026-09-19)

Pergunta do orientando sobre a seleção de landmarks, o eixo mais trabalhado no Nyström-LSSVM.
Script reproduzível: `scripts/study_colnorm_criterion.py`; dados em `results/colnorm_criterion.json`.

### Não há assimetria entre os dois modelos

Tanto o Nyström-LSSVM quanto o FT-CUR usam `colnorm` = amostragem ∝ ‖x_i‖² no espaço de **entrada**.
Em `nystrom.py` a escolha é explícita: só `leverage` recebe o kernel, justamente "to avoid computing the
full N×N kernel matrix (O(n²) memory)". Minha suposição inicial de que o LSSVM usava a matriz kernel
verdadeira estava **errada**.

### O critério é anticorrelacionado com o que o nome promete

Spearman entre ‖x_i‖² e ‖K[:,i]‖² (a norma de coluna de Drineas), 10 datasets do Tier 1:

| σ | ρ com a norma de coluna | ρ com o leverage de posto m |
|---|---|---|
| 0,1 | −0,28 | −0,17 |
| 0,5 | −0,67 | −0,20 |
| 2,0 | −0,96 (8/10 abaixo de −0,9) | **+0,80** |
| 5,0 | −1,00 (9/10) | **+0,82** |
| 8,0 | −1,00 (10/10) | **+0,82** |

Faz sentido geométrico: com RBF e dados padronizados, ‖x_i‖ grande = ponto periférico = longe de todos =
coluna de kernel de norma **pequena**. Os dois critérios são opostos.

**O regime importa, e é favorável ao achado.** O σ que o GridSearchCV seleciona: Tier 1 modal 8,0 (43%),
depois 0,5 (30%) e 2,0 (17%); Tier 2 **100% em 5,0**. Ou seja o Tier 2 está inteiramente no regime em que
ρ ≈ −1,00, e no Tier 1 a maioria (σ ≥ 1,5, ≈68%) também. A ressalva honesta é o σ = 0,5 do Tier 1, onde
a anticorrelação é moderada (−0,67) e a correlação com leverage **desaparece** (−0,20); e em σ = 0,1 o
kernel é quase a identidade e nada se mede.

### Duas consequências para o texto

1. **A nota de rodapé de `Capitulo3.tex:82` pode passar de hedge a fato medido.** Hoje ela diz que ‖x_i‖²
   "não coincide" com a norma de coluna da matriz derivada. Não é questão de não coincidir: para σ ≥ 2
   é praticamente o **inverso** (ρ = −1,00).
2. **A mesma nota precisa ser corrigida num ponto.** Ela afirma que o critério "não é uma aproximação dos
   *leverage scores*". Para σ ≥ 2 ele é justamente isso, e bom: ρ ≈ +0,82. Ambos selecionam pontos
   atípicos. Isso responde em parte a lacuna que o próprio `Apendice1.tex:347` declara (ausência de
   comparação empírica com leverage): no regime de operação, `colnorm` **já é** um substituto barato de
   leverage, com ρ ≈ +0,8, sem pagar a SVD.

### E explica o resultado nulo, que já está na tese

`Apendice1.tex:319` reporta que nos dois regimes de escassez o `colnorm` é o **pior** seletor. Agora há
mecanismo: ele é um proxy de leverage, e a análise de \citeonline{zhang2008improved} — já citada no
mesmo apêndice — diz que o que governa o erro de Nyström é o erro de **quantização** (k-means), não a
importância espectral (leverage). Um critério tipo-leverage é, portanto, *previsto* a não ajudar. O nulo
deixa de ser coincidência e passa a ser consequência.

Para o FT-CUR o argumento é mais forte ainda, e o `Apendice1.tex:331` já aponta na direção certa: a
matriz-alvo é a atenção **aprendida**, que não existe antes do treino e muda a cada passo, logo nenhum
critério fixo no espaço de entrada pode ser o correto. Coerente com a ablação: random 0,7447, colnorm
0,7435, kmeans 0,7432, opposite 0,7405 — e colnorm − random = −0,0012 com p = 0,83.

### Oportunidade barata (opcional)

`ColumnNormSelector` já aceita `kernel=`. Passar o kernel implementa a norma de coluna **verdadeira**, e
em N ≤ 5000 a matriz K é trivial (25M floats). Isso permitiria responder, com um seletor a mais na
ablação já planejada, se o critério fiel de Drineas bate o random — pergunta que hoje a tese responde
apenas por citação de terceiros. Custo estimado: ~2,5 h de ajuste (1,25 h em 2 placas) no Tier 1.

### FT-CUR em mini-lote: o seletor importa, e o `colnorm` é o pior (2026-09-19)

`scripts/study_ftcur_minibatch_selection.py`, dados em `results/ftcur_minibatch_selection.json`.
Mini-lote forçado (`batch_size=512` contra `n_fit=1120`), `m_ratio` fixo em 10%, 4 datasets do Tier 2
× 5 sementes = 20 pares. Inclui o novo seletor `colnorm_inv` (∝ 1/‖x_i‖²), que reproduz a norma de
coluna verdadeira do kernel em O(nd).

| braço | F1 | vs per_batch | vs global+random |
|---|---|---|---|
| per_batch (histórico; landmarks do fit são DESCARTADOS) | 0,7309 | — | |
| global + random | 0,7309 | +0,0000 (p=0,96) | — |
| global + colnorm_inv | 0,7248 | −0,0061 (p=0,25) | −0,0061 (p=0,20) |
| global + colnorm | 0,7170 | −0,0139 (p=0,044) | −0,0140 (p=0,071) |

Friedman entre os quatro: p = 0,060. Sob Holm, o p = 0,044 não sobrevive à multiplicidade — é sinal de
direção, não prova.

Três leituras:

1. **Reamostragem não traz nada.** `per_batch` (sorteio novo a cada lote) e `global+random` (conjunto
   fixo) empatam exatamente. Uma leitura parcial com apenas BANK e TELCO sugeria o contrário; com os 20
   pares a diferença é zero.
2. **Inverter conserta o dano, mas o teto é o sorteio uniforme.** `colnorm_inv` supera o `colnorm` em
   +0,0078 (12/20), o que devolve o critério ao empate com o random, e não o faz superá-lo.
3. **O `colnorm` é o único braço abaixo da linha de base.** Coerente com as outras três vias medidas
   hoje: empate dos quatro seletores em lote completo (colnorm −0,0012, p=0,83); em reconstrução de
   Nyström o critério fiel não bate o que está em uso e nenhum bate o k-means; e aqui ele é o pior.

**O que isso muda na tese.** Nada no corpo: Tier 1, Tier 2 e Ablação D rodam todos em lote completo
(n_fit máximo 4.000 contra `batch_size=4096`), então ali o `colnorm` é aplicado de verdade e o resultado
é o empate já publicado. Muda a **Tabela 19**, que roda em mini-lote com N até 50.000: lá, no caminho
`per_batch`, o seletor descrito no texto nunca é usado (sorteio uniforme dentro do lote), e no caminho
`global` ele é a pior das quatro opções.

**Correção pendente no código (não aplicada):** a Ablação D passa por 96 amostras de folga
(n_fit = 4.000 contra batch 4.096). Se `val_fraction` ou N mudarem, ela cruza para o caminho de
mini-lote sem aviso. Vale uma asserção ou um log quando o caminho de mini-lote for acionado.

#### k-means no mini-lote: nenhum ganho de F1 (2026-09-19)

Acrescentado o braço `global+kmeans` (critério de quantização de Zhang 2008, `n_init=10,
max_iter=300`, os defaults que o estudo usa) aos mesmos 20 pares. Resultado:

| braço | F1 | vs per_batch | vs global+random |
|---|---|---|---|
| per_batch | 0,7309 | — | |
| global + random | 0,7309 | +0,0000 (p=0,96) | — |
| global + kmeans | **0,7308** | **−0,0001** (p=0,65) | **−0,0001** (p=0,33) |
| global + colnorm_inv | 0,7248 | −0,0061 (p=0,25) | −0,0061 (p=0,20) |
| global + colnorm | 0,7170 | −0,0139 (p=0,044) | −0,0140 (p=0,071) |

Friedman entre os cinco: p = 0,156.

**Não há ganho a comprar, logo a discussão de custo é vazia.** Fica registrado o que se apurou dela, caso
volte à pauta: (a) seleção POR LOTE é O(N·B·d·iters) por época, linear em N, mas o custo se acumula por
época, e o total é a seleção global vezes B·E/N — só compensa para N > B·E (~20.480 com B=512, E=40);
medido, é 30× mais caro em N=2000 e 34× em N=5000; (b) os 85 s do k-means em N=16.000 no
`app_selection_cost.tex` vêm dos defaults do `KMeansSelector` (`n_init=10, max_iter=300`); com
`n_init=1, max_iter=25` cai para ~3,3 s em N=20.000. O número da tese está correto para a implementação
que ela usa, mas o custo é escolha de parâmetro, não propriedade do método.

**Quarta confirmação independente** de que a identidade dos landmarks não limita o FT-CUR: (i) empate dos
quatro seletores em lote completo (colnorm −0,0012, p=0,83); (ii) em reconstrução de Nyström o k-means é
o único que melhora, e essa vantagem não se transfere para classificação; (iii) no mini-lote o
`colnorm_inv` apenas recupera o empate com o sorteio; (iv) o k-means aplicado de verdade no mini-lote
rende −0,0001. O gargalo é o próprio m ≪ N.

Escopo do estudo: 4 datasets do Tier 2 × 5 sementes, N=2000, B=512, m_ratio=10%. Direção consistente com
tudo o mais, mas é estudo pequeno.

#### Nenhuma região da distribuição de norma é privilegiada (2026-09-19)

Ideia do orientando: se as duas caudas de ‖x_i‖² dão o mesmo, o miolo também dá? Dois seletores novos —
`complement` (sorteio no que `colnorm` e `colnorm_inv` NÃO pegaram, sobreposição zero com ambos) e
`midband` (descarta os quartis extremos, sorteia nos 50% centrais). Mesmos 20 pares:

| braço | F1 | Δ vs random | p |
|---|---|---|---|
| per_batch (resorteio por lote) | 0,7309 | −0,0000 | 0,96 |
| global + random | 0,7309 | — | — |
| global + kmeans (quantização) | 0,7308 | −0,0001 | 0,33 |
| global + complement | 0,7266 | −0,0044 | 0,31 |
| global + midband | 0,7266 | −0,0044 | 0,84 |
| global + colnorm_inv (miolo denso) | 0,7248 | −0,0061 | 0,20 |
| global + colnorm (periferia) | 0,7170 | −0,0140 | 0,071 |

Friedman entre os sete: p = 0,210. **Amplitude entre melhor e pior: 0,0140.** Norma média dos landmarks
escolhidos, para dimensionar o quanto os braços diferem de fato: colnorm 28,9 | random 14,8 |
complement 14,5 | midband 11,9 | colnorm_inv 11,1 (média geral 16,0).

**Consequência para o texto.** A afirmação hoje é que *o seletor escolhido* é imaterial, o que se pode
ler como "testaram quatro e empataram por sorte". Com isso ela pode passar a ser mais forte e mais
defensável: **a posição do landmark no espaço de entrada é imaterial** — varreu-se a distribuição inteira
de ‖x_i‖² (periferia, miolo, faixa central, complemento das caudas), mais o critério de quantização, mais
o sorteio uniforme, mais o resorteio por lote, e nada se distingue. O mecanismo é o já descrito em
`Apendice1.tex:331`: a matriz-alvo é a atenção APRENDIDA, sem relação fixa com a geometria de entrada,
logo nenhum critério de entrada pode ajudar por construção.

Escopo: 4 datasets do Tier 2 × 5 sementes, N=2000, B=512, m_ratio=10%, mini-lote forçado. Nada aqui
contradiz o corpo da tese, que roda em lote completo; o que muda é o alcance do que se pode afirmar.

##### RESSALVA a corrigir no item acima

Os quatro datasets do estudo de mini-lote (BANK, TELCO, ADULT, SHOPPERS) são todos **tabulares reais** —
exatamente o regime em que `Apendice1.tex:319` já estabelecia que a seleção é irrelevante ("nos seis
datasets tabulares reais, a seleção continua irrelevante mesmo a m/n=5%"). Onde a seleção PASSA a
importar é nos três sintéticos de estrutura geométrica 2D, e é lá que o k-means ganhou +0,035
significativos no Nyström-LSSVM a m/n=10%.

Portanto a conclusão "nenhuma região da distribuição de norma é privilegiada" está estabelecida apenas
para tabular real, e não generaliza sem teste nos sintéticos. Reexecução dos sete braços em TWS_2k,
TWM_2k e TWC_2k (mesmo N=2000, mesmo B=512, mesmo m_ratio, só o tipo de dado muda) em
`results/ftcur_minibatch_geo.json` — ver resultado no item seguinte.

##### Resultado nos sintéticos: o midband NÃO se confirma; e o k-means inverte de sinal

Sete braços em TWS_2k/TWM_2k/TWC_2k (5 sementes) e depois TWC_2k com **30 sementes**
(`results/ftcur_minibatch_geo.json`, `results/ftcur_minibatch_twc30.json`).

Nos três sintéticos com 5 sementes a amplitude foi 0,0864 contra 0,0140 dos tabulares reais, e o
`midband` aparecia em primeiro com 0,9652. Mas TWM e TWS estão **saturados** (0,98 a 1,00), então todo o
sinal vinha do tabuleiro. Com 30 sementes em TWC_2k:

| braço | F1 | dp | Δ vs random | p |
|---|---|---|---|---|
| global + random | **0,8246** | 0,168 | — | — |
| global + midband | 0,7893 | 0,197 | −0,0353 | 0,94 |
| global + complement | 0,7570 | 0,199 | −0,0675 | 0,60 |
| per_batch | 0,7544 | 0,192 | −0,0702 | 0,27 |
| global + colnorm | 0,7475 | 0,199 | −0,0771 | 0,28 |
| global + colnorm_inv | 0,7329 | 0,210 | −0,0916 | 0,094 |
| global + kmeans | 0,7326 | 0,196 | −0,0919 | **0,0067** |

Friedman: p = 0,158. **O `midband` caiu de 0,9159 para 0,7893** ao ir de 5 para 30 sementes — o desvio
padrão nesse dataset é 0,20, e a estimativa de cinco sementes errou por 0,13. Era variância; a ideia não
se sustenta e NÃO deve virar seletor.

**Dois resultados que ficam:**

1. **O sorteio uniforme é o melhor braço** (0,8246, acima dos seis outros), o que reforça a decisão de
   adotá-lo como seletor principal em lugar do `colnorm`.
2. **O k-means é significativamente PIOR que o sorteio** no tabuleiro (−0,0919, p = 0,0067, 9/30, único
   a sobreviver a Holm sobre as seis comparações). Isso **inverte** o que `Apendice1.tex:319` reporta para
   o Nyström-SVM, onde o k-means ganhava +0,035 significativos nos sintéticos a m/n = 10%. Mecanismo: no
   LSSVM o alvo é o kernel RBF fixo, e minimizar erro de quantização de fato ajuda; no FT-CUR o alvo é a
   atenção APRENDIDA, e agrupar o espaço de entrada concentra landmarks nos centros dos quadrantes —
   justamente os pontos menos informativos sobre a fronteira. Os dois modelos respondem em sentidos
   opostos ao mesmo seletor, o que é uma distinção genuína entre aproximar kernel fixo e aproximar
   atenção aprendida, e merece um parágrafo no apêndice.

##### A FAZER: testar o `midband` no Nyström-LSSVM

Pedido do orientando (2026-09-19). O `midband` é o **segundo melhor** braço em toda parte no FT-CUR —
0,7893 no tabuleiro (atrás só do sorteio, 0,8246) e 0,7266 nos tabulares reais — e nunca foi testado no
Nyström-LSSVM. Lá o argumento a favor dele é bem mais forte que no FT-CUR, por três razões:

1. A matriz-alvo é **fixa**: o kernel RBF existe antes do treino, então um critério de entrada pode ter
   relação estável com ela — ao contrário da atenção aprendida, onde o canal se rompe.
2. É o regime em que a teoria de quantização de \citeonline{zhang2008improved} de fato se aplica, e em
   que o k-means **ganha** (+0,035 nos sintéticos a m/n = 10%) em vez de perder como no FT-CUR
   (−0,0919, p = 0,0067, no tabuleiro).
3. Custo idêntico ao `colnorm` e menor na prática: O(nd) nos dois, medido 2,94 ms contra 3,60 ms em
   N = 20.000 (e 7,06 contra 8,27 ms em N = 50.000). Contra o k-means, que custa 85 s em N = 16.000.

Desenho sugerido: acrescentar `NystromLSSVMMidband` ao roster e rodar nos três regimes da ablação já
existente (Tier 1 m/n = 30%, escassez m = 5% e 10%), 30 sementes, comparando contra `random` como linha
de base. É CPU, então é barato. A pergunta específica: em estrutura de variedade 2D, a faixa central da
distribuição de norma supera o sorteio uniforme — isto é, existe posição privilegiada quando a
matriz-alvo é fixa? O `midband` já está implementado (`landmark_selection.MidBandSelector`, chave
`'midband'`), então falta só o wrapper e o runner.

Nota de método: se der positivo, é achado de trabalho futuro, **não** motivo para trocar o seletor do
corpo principal — a troca já decidida é para `random`, que é a opção simples justificada pela
imaterialidade, e trocar depois pelo vencedor de uma comparação posterior seria seleção sobre o próprio
resultado.
