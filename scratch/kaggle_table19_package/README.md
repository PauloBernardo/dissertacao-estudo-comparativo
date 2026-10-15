# Pacote para Kaggle — atualização da Tabela 19

## O que é

Roda todos os modelos que já estão na Tabela 19 da dissertação, medindo
**fit e predição separadamente** (memória e tempo) — a Tabela 19 publicada
só tinha essa distinção para o fit; a predição nunca foi medida à parte,
mesmo o script oficial já vindo preparado pra isso desde o commit `134d79f`.

Inclui também a correção do FT-CUR: o `_logits` publicado tem um bug
**estrutural** (não uma opção de design) — a predição de um ponto de teste
depende de quais outros pontos de teste estão no mesmo lote, e a memória de
predição cresce sem limite com N (OOM verificado em N=50.000, localmente).
`FTCUR_minibatch` e `FTCUR_mfixed_full` neste pacote já são as versões
corrigidas (o bug independe de como o treino é *batchado* — se corrige um,
corrige os dois) — são os números que devem ir pra tabela final, não
alternativas a escolher.

**Isto NÃO é a mesma coisa que a discussão de `m_ratio` vs. `m` fixo.**
Esse benchmark sempre usou `m` FIXO (`M_MINIBATCH=64`, `M_FIXED=256`), o
regime genuinamente sub-quadrático `O(N·m)` — nunca teve o problema (que
existe em outro lugar, na grade `m_ratio` do Tier1/2) de a complexidade
voltar a ser `O(r·N²)`. O bug aqui é de IMPLEMENTAÇÃO, não de escolha de
hiperparâmetro: mesmo com `m` fixo, o código antigo processava a predição
inteira numa chamada só, sem *chunk*, sobre `[X_train ‖ X_test]` junto.

## Arquivos

- `run_table19_benchmark.py` — roda o benchmark. Acha a raiz do repo sozinho
  (procura `src/models/ft_transformer_model.py` subindo a partir de onde o
  script está); se não achar, use `--repo-root`.
- `make_table19.py` — lê o JSON de resultado e gera a tabela LaTeX pronta
  (fit(s) / fit mem(MB) / pred(ms) / pred mem(MB), sem coluna de total).

## Como rodar no Kaggle

0. **Settings → Accelerator → GPU** (T4 costuma bastar) antes de rodar
   qualquer célula — sem isso o script roda em CPU e fica muito mais lento
   (e os números de VRAM saem zerados).
1. Clonar/subir o repo `sparse-lssvm-transformers-study` inteiro (precisa de
   `src/`) num notebook com GPU ativada.
2. Copiar os dois scripts deste pacote pra dentro do repo (em `scripts/`,
   por exemplo — tanto faz o lugar exato, o script acha a raiz sozinho).
3. Rodar o benchmark (leva um tempo — 11 variantes × 7 N's × 3 repetições,
   com corte automático em OOM; **é resumível** -- se a sessão cair, rodar
   de novo com o mesmo `--output` continua de onde parou, sem recomputar o
   que já foi salvo):

   ```bash
   python run_table19_benchmark.py --output table19_results.json
   ```

   Pra rodar só um subconjunto primeiro (teste rápido):

   ```bash
   python run_table19_benchmark.py --variants SAINT_minibatch,FTCUR_minibatch \
       --output table19_quick.json
   ```

4. Gerar a tabela LaTeX:

   ```bash
   python make_table19.py --input table19_results.json --output tabela19.tex
   ```

   Por padrão usa N = 1.000/5.000/10.000/20.000 (os mesmos 4 pontos da
   Tabela 19 atual) e **não inclui** as linhas de referência do bug
   (`FTCUR_minibatch_bug_reference`, `FTCUR_mfixed_full_bug_reference`) —
   elas existem só como evidência de comparação, não são candidatas a virar
   linha da tabela final. Pra ver o "antes vs depois" lado a lado (bom pro
   texto/nota de rodapé, não pra tabela oficial):

   ```bash
   python make_table19.py --input table19_results.json --output tabela19_com_antes.tex \
       --include-bug-reference
   ```

   Pra escolher outros N's (o JSON tem 500/1000/2000/5000/10000/20000/50000):

   ```bash
   python make_table19.py --input table19_results.json --n-values 2000,10000,50000
   ```

5. Baixar `table19_results.json` (dado bruto, guardar) e `tabela19.tex`
   (colar em `dissertacao-latex/tables/benchmark_transformers_memory.tex`).

## O que esperar ver

Já validado localmente (GPU MX350, 2GB) até N=50.000:

| | SAINT (mini) | FT-CUR **antes** | FT-CUR **corrigido** |
|---|---|---|---|
| VRAM predição, N=5.000 | ~24 MB | 172 MB | 24 MB |
| VRAM predição, N=20.000 | ~26 MB | 596 MB | 24 MB |
| VRAM predição, N=50.000 | ~28 MB | **OOM** | 24 MB |

A correção deve reproduzir esse padrão em qualquer hardware — VRAM de
predição do FT-CUR corrigido constante, independente de N; o "antes" cresce
sem limite. Fit e F1 não são afetados pela correção (validado em sessões
anteriores: Tier 1 completo 300/300, sanity checks com diff < 1e-6).

A comparação que isola de fato o efeito da correção é `SAINT_minibatch` vs.
`FTCUR_minibatch` (as duas com treino igualmente barato e limitado, só muda
o comportamento da predição). **`*_fullbatch`/`*_mfixed_full` NÃO são
comparáveis entre si**: "full" desliga o *mini-batch* inteiramente, e isso
custa coisas muito diferentes pra cada arquitetura -- SAINT vira atenção
densa `O(N²)` de verdade (aqui já morre no FIT, OOM em N=10.000 no MX350,
antes até de chegar na predição), enquanto o FT-CUR continua `O(N·m)`
mesmo "sem lote" porque a aproximação por *landmarks* protege ele
independente de como o treino é batchado. Não tratar o par *_fullbatch*
como um teste de "quem é mais eficiente" -- são dois piores-casos
diferentes, cada um específico da própria arquitetura.

**`FTCUR_mfixed_full`, mesmo corrigido, ainda cresce e eventualmente OOMa no
FIT** (não mais no predict) -- verificado localmente: 110MB (N=1.000) →
485MB (N=5.000) → 948MB (N=10.000) → OOM (N=20.000, MX350). Isso é
esperado e **não é** o bug corrigido: `batch_size=None` significa treino
sem nenhum *chunk*, então o custo linear em N (landmarks fixos, mas
embeddings/ativações de N linhas por época) é inerente à própria definição
de "full-batch", presente igualmente antes e depois da correção (os
valores de `fitVRAM` de `FTCUR_mfixed_full` e `FTCUR_mfixed_full_bug_reference`
são praticamente idênticos em qualquer N). No Kaggle (GPU maior) o ponto de
OOM deve ficar mais alto, mas o crescimento linear e o eventual estouro
devem se repetir -- não é regressão da correção, é característica da
variante.

## Avisos

- Datasets sintéticos (`sklearn.make_classification`), sem F1 nesta medição
  — é benchmark de eficiência, não de qualidade preditiva. A evidência de
  F1 preservado vem de outros testes (Tier 1 completo, sweep em dados
  reais), não deste script.
- A correção aqui é sobre um bug de *implementação* (predição não-*chunked*,
  acoplada entre pontos de teste). Não confundir com a questão separada de
  `m_ratio` vs. `m` fixo na grade oficial do Tier1/2 (`src/tuning/grids.py`)
  — esse benchmark sempre usou `m` fixo, então não testa nem resolve aquilo.
