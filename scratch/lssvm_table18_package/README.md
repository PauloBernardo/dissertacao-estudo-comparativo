# Pacote — atualização da Tabela 18 (família LSSVM)

## O que é

Extensão de `scripts/run_scaling_benchmark.py` (o gerador atual da Tabela
18) em duas frentes, nascidas da pergunta: *"o PCP e o OppMaps têm a maior
esparsidade da Tabela 17 — isso deveria significar menos memória na
predição, mas eles nem aparecem na Tabela 18."*

1. **Fit e predição medidos separadamente** (RAM de pico por fase) — a
   Tabela 18 publicada só tem uma coluna "Mem.", sem essa distinção é
   impossível ver o efeito da esparsidade, que só deveria aparecer na
   *predição* (menos vetores de suporte → kernel(teste, suporte) menor).
2. **Adiciona os 4 modelos que faltavam**: PCP, IP, Pruning, OppMaps — o
   script atual só media as 5 variantes ADMM/Nyström/denso.

**Verificação prévia importante (rodada antes de escrever o benchmark):**
`_kernel_predict` (`src/models/lssvm/base.py`) **faz *subsetting* real** —
`K = self.kernel_matrix(X, self.X_train_[nz])`, só os vetores de suporte
não-nulos. Ao contrário da esparsidade de atenção dos Transformers
(numérica, não algorítmica — zeros armazenados e multiplicados como
qualquer entrada), aqui a esparsidade **genuinamente** reduz o trabalho na
predição. A hipótese é testável de verdade.

## Hiperparâmetros dos 4 modelos novos — validados, não chutados

O benchmark anterior (`OLD/scripts/run_lssvm_efficiency_benchmark.py`,
**não usado aqui** — implementações antigas/adaptadas, descartado) testava
o PCP com uma config que dava `sparsity_ratio=0.000` (idêntico ao denso) —
invalidando qualquer leitura sobre memória de predição.

Os hiperparâmetros usados neste pacote vêm da **moda das configurações
vencedoras no Tier 2 real** (`results/tier2_gridcv.json`), e reproduzem a
Tabela 17 quase exatamente:

| modelo | config | esparsidade alvo (Tabela 17) | reproduzido |
|---|---|---|---|
| PCP | `sigma=5.0, tau=0.1, rank=200` | 0,907 | 0,9068 |
| IP | `sigma=5.0, tau=12.5, selection_ratio=0.7` | 0,518 | 0,5183 |
| Pruning | `sigma=5.0, tau=12.5` (+ `drop_tolerance=0.05, pruning_rate=0.05`) | 0,542 | 0,5417 |
| OppMaps | `sigma=5.0, tau=2.5, proto_ratio=0.7` | 0,839 | 0,8391 |

(No benchmark sintético deste pacote os valores exatos variam um pouco —
dataset diferente, e o `rank` do PCP é uma contagem absoluta, não uma
razão, então a compressão cresce com N por construção. Isso é esperado,
não é erro.)

## Arquivos

- `run_table18_benchmark.py` — roda o benchmark (CPU, todos os 9 modelos).
  Acha a raiz do repo sozinho.
- `make_table18.py` — gera a tabela LaTeX (fit/pred separados, sem total,
  mais uma coluna de esparsidade média por modelo).

## Rodando

CPU, não precisa de Kaggle/GPU — já está rodando localmente em background
nesta sessão (`table18_results.json`, resumível, mesmo padrão do pacote da
Tabela 19).

```bash
python run_table18_benchmark.py --output table18_results.json --repeats 3
python make_table18.py --input table18_results.json --output tabela18.tex
```

Pra um subconjunto:
```bash
python run_table18_benchmark.py --variants PCPLSSVm,OppositeMapsOriginalLSSVM \
    --output table18_quick.json
```

## Cuidado: CPU não tem "OOM" avisando — os N_MAX são calibrados por tempo real

Diferente do pacote da Tabela 19 (GPU, `torch.cuda.OutOfMemoryError` corta
automaticamente), aqui não existe sinal de estouro — um treino lento
simplesmente trava o script sem avisar. Medi antes de fixar os limites:

| modelo | tempo observado | N_MAX escolhido |
|---|---|---|
| PCP | 1,6s em N=5.000 | 20.000 (escala bem) |
| OppMaps | 9,6s em N=5.000 (~O(N^2,1)) | 20.000 (~9min/rep em N=20k) |
| IP | 11,6s em N=2.000 (~O(N^2,7)) | **5.000** (N=10k já passaria de 1h) |
| Pruning | 23,3s em N=2.000 (~O(N^2,5)) | **5.000** (idem) |

IP e Pruning **não salvam nada até terminar as `--repeats` de um N** (ao
contrário do resto do script, que salva incrementalmente) — por isso os
limites ficaram conservadores. Se quiser estender pra N maior nesses dois,
espere um tempo bem mais longo e sem feedback intermediário.

## O que esperar ver

Se a hipótese (esparsidade → menos memória/tempo de predição) se confirmar,
o padrão deve ser: `PCP` (maior esparsidade, cresce com N) e `OppMaps`
devem mostrar `predRAM`/`pred_s` crescendo **mais devagar** que
`StandardLSSVM` (denso, 0% esparsidade) conforme N aumenta — o kernel de
predição, no denso, sempre usa as N linhas inteiras; nos esparsos, só os
`n_support_` que sobraram. `IP` e `Pruning`, com esparsidade menor e mais
constante (~30-64%), devem ficar num meio-termo.
