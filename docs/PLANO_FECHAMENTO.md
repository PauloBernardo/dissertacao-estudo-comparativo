# Plano de fechamento da dissertação — versão 1 (protocolo publicado) e versão 2 (protocolo dos artigos)

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
7. **Piloto do protocolo dos artigos lançado** (19:53, CPU, 3 sementes, TWS/HAB/AI4I/BANK/TELCO): saída incremental em `results/pilot_transformer_budget.json`, log em scratchpad da sessão. Ver §3.

## 2. VERSÃO 1 — fechar com implementações fiéis, protocolo publicado, limitação declarada
- [ ] **1.1** Kaggle: concluir as 6 fases do notebook; baixar os JSONs para `results/`.
- [ ] **1.2** `bash scripts/post_rerun_saint_entmax.sh` (faz merge, regenera, copia, normaliza, Nemenyi, reaplica edições manuais, compila). Conferir o resumo de diffs que ele imprime.
- [ ] **1.3** Texto — números de SAINT e FT-Entmax: Cap. Resultados (Tier 1 bullets e Friedman; §esparsidade Transformers; Tier 2 bullets, esparsidade e métricas; Ablação D §SAINT; benchmark inter-instâncias), Conclusão (itens 1, 2, 5, 6), Apêndice (Nemenyi, métricas complementares), Resumo/Abstract só se alguma conclusão mudar.
- [ ] **1.4** Texto — Ablação A inteira dos Transformers (agora por transferência; os seis mudam). Reescrever o parágrafo da espiral/SAINT e a decomposição por semente com os dados novos.
- [ ] **1.5** Texto — limitação do orçamento de treino: (a) Metodologia §Ambiente/Transformers: uma frase com os passos efetivos por tier e a referência aos regimes dos artigos; (b) Limitações: item novo; (c) Trabalhos futuros: protocolo em passos alinhado aos artigos (= versão 2).
- [ ] **1.6** Apêndice curto "Ablação de orçamento de treino": tabela espiral/HAB × {protocolo, 40 ép. lote 32, 400 ép.} para FT-Softmax e SAINT (dados desta sessão; ideal repetir em GPU com 10 sementes nos 3 sintéticos e nos 6 modelos — ~1 h de GPU).
- [ ] **1.7** Texto — Ablação D: parágrafo de validação por re-tuning do FT-CUR no BANK (item 1.6 acima do §1), espelhando o do ADMM-Nyström; qualificar "SAINT é o que mais se beneficia de N" (depende do BANK; sobrevive ao re-tuning nesse dataset).
- [ ] **1.8** Texto — suavizar: "data-hungry" → "sob orçamento fixo de 40 épocas"; "FT-CUR empata com SAINT" → "sob o mesmo orçamento"; verbos do Resumo/Conclusão em linha com o poder do Nemenyi (CD ≈ 9,4).
- [ ] **1.9** Itens do juízo de valor ainda abertos: parágrafo "o que fica de meu" (Intro + Conclusão); limitação do orçamento de tuning assimétrico (LSSVM 36–75 configs vs Transformers 6); título ("formulações duais" no plural); Folha de Aprovação (membros); parágrafos longos da Discussão; apêndice de reprodutibilidade (commit + mapa JSON→tabela + script).
- [ ] **1.10** Nota no Cap. 5: variantes de mini-lote da Tabela 19 são configurações de custo, sem F1 avaliado; rota streaming verificada equivalente (Δ ≤ 0,008).
- [ ] **1.11** Recompilar; conferir log (0 overfull, 0 undefined); memória do projeto atualizada; commit + push (dados, tabelas, docs).

## 3. VERSÃO 2 — re-executar os seis Transformers sob o protocolo dos artigos (decisão depois do piloto)
Protocolo proposto (`PROTOCOLS["paper"]` em `scripts/pilot_transformer_budget.py`):
AdamW, lr 1e-4, sem agenda; lote 256; parada após 16 épocas sem melhora na validação, **piso de 200 épocas** (única regra nossa: nos datasets em que 1 época = 1 passo), teto 1000; melhor ponto de validação; decaimento de peso por artigo (1e-5 FT, 0,01 SAINT); arquiteturas e grades como estão. FT-CUR: decidir entre lote completo (desenho original) e mini-lote com landmarks globais (`minibatch_landmarks="global"` + `predict_mode="streaming"`).
- [x] **2.1** Piloto — CONCLUÍDO 2026-09-18 20:47 (ver §3.1/3.2 abaixo).
- [ ] **2.1-bis** Piloto na CPU: `python scripts/pilot_transformer_budget.py --datasets TWS HAB AI4I BANK TELCO --seeds 3` (≈3–4 h). Responde: o piso tira os modelos do platô? lr 1e-4 basta? FT-CUR mini-lote global se sustenta em F1? custo por ajuste → horas de GPU.
- [ ] **2.2** Decidir alcance: (a) seis Transformers em tudo (Tier 1, Tier 2, Ablações A–D, Tabela 19): estimativa 120–200 h de GPU (publicado consumiu ≈40 h: 10,6 Tier 1 + 17,8 Tier 2 + 5,9 Abl. A + 4,6 Abl. B/C + 1,2 N=5000); (b) só SAINT e FT-CUR nos Tiers + ablação de orçamento nos sintéticos para os seis: ≈30–40 h.
- [ ] **2.3** Implementar o perfil de protocolo em `src/tuning/grids.py` (novo conjunto de `fixed`), notebook Kaggle por fases (copiar o esquema do atual), rodar.
- [ ] **2.4** Merge em JSONs NOVOS (não sobrescrever os da versão 1), regerar em uma cópia da tese, comparar as conclusões; então decidir se a versão 2 substitui a 1 ou entra como capítulo/apêndice.
- [ ] **2.5** Se substituir: revisar todo o Cap. 5 e a Conclusão; se não: registrar como trabalho futuro com os números do piloto.

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

| Opção | O que muda | GPU estimada | Kaggle (30 h/semana) |
|---|---|---|---|
| A | tudo como a v1: grade + 5 folds + 30 sementes, seis modelos | ≈ 560 h | 19 semanas — inviável |
| B | idem com 10 sementes | ≈ 190 h | 6–7 semanas |
| C | **configuração fixa por modelo (padrão dos artigos / moda da v1), sem grade**, 30 sementes, seis modelos | ≈ 15–20 h | 1 semana |
| D | só SAINT e FT-CUR com grade + 5 folds, 30 sementes | ≈ 45 h (SAINT) + 80–145 h (FT-CUR) | 4–6 semanas |
| E | C + grade só no par SAINT/FT-CUR no Tier 2 | ≈ 60–80 h | 2–3 semanas |

Recomendação: **C** (é o desenho do artigo do FT-Transformer: "default configuration performs on par with tuned"), declarando que na v2 os Transformers usam configuração fixa enquanto os LSSVMs mantêm a grade — assimetria oposta à atual (hoje a grade dos Transformers tem 6–12 configurações contra 36–75 dos LSSVMs) e menos grave, pois o que a v2 quer medir é o efeito do orçamento de treino. Abl. A já é por transferência (barata). Decisão pendente.
