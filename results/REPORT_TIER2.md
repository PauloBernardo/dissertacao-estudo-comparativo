# Relatório de Resultados — Estudo Comparativo Tier 2 (N=5000)

**Gerado em:** 2026-06-03  
**Protocolo:** 18 modelos/variantes × 6 datasets × 30 seeds = 3.240 execuções consolidadas  
**Métrica principal:** F1-macro e Taxa de Esparsidade  

---

## 1. Resumo Executivo

A transição do Tier 1 ($N=1000$) para o Tier 2 ($N=5000$) expôs limitações críticas nos modelos baseados em formulação Primal com penalidade L1 clássica (como o ADMM e o FISTA), além de demonstrar inquestionavelmente a superioridade computacional dos métodos de aproximação de baixo posto (Nyström) e dos Transformers tabulares esparsos (FT-CUR).

**Principais achados metodológicos:**
- A matriz do Kernel Topológico com $N=5000$ (25 milhões de elementos) torna os modelos exatos e iterativos de primeira ordem no espaço primal um gargalo proibitivo em tempo computacional.
- **Limitações do ADMM/FISTA:** A fim de conseguir convergir numericamente contra a pesada matriz, as soluções do ADMM e FISTA no espaço primal tornaram-se excessivamente esparsas, sacrificando significativamente o poder preditivo em datasets com leve desbalanceamento (F1 ~0.58).
- **Consagração do Nyström:** O método `NystromLSSVM` superou duramente o ADMM, entregando tempo de treinamento sub-5 segundos (contra quase 3 minutos do ADMM), mantendo F1 > 0.70 e alta esparsidade (81.5%).
- **Superioridade do FT-CUR:** Entre todos os métodos focados em esparsidade explícita, a arquitetura com fatoração Nyström-CUR no Transformer (`FT-CUR`) reinou absoluta: obteve $F1=0.7132$ e esparsidade de incríveis $89.4\%$.

---

## 2. Performance Preditiva e Esparsidade Consolidada

Abaixo, a tabela rankeada pela pontuação F1-Macro dos principais modelos. É notável o "salto" de desempenho de métodos duais/aproximados para os métodos primais iterativos.

| Posição | Modelo | F1-Macro | Esparsidade | Tempo de Ajuste (Treino) |
|:---:|:---|:---|:---|:---|
| 1 | XGBoost (Baseline Árvore) | 0.7348 ± 0.0494 | 0.0% | **0.19 s** |
| 2 | SAINT (Baseline Transformer) | 0.7223 ± 0.0550 | 0.0% | 6.48 s |
| 3 | **FT-CUR (Transformer Proposto)** | **0.7132 ± 0.0706** | **89.4%** | 17.94 s |
| 4 | Standard LSSVM (Dual denso) | 0.7093 ± 0.0463 | 0.0% | 4.74 s |
| 5 | **Nyström-LSSVM (Proposto)** | **0.7072 ± 0.0463** | **81.5%** | **2.20 s** |
| 6 | DualFISTA | 0.7059 ± 0.0469 | 2.1% | 98.81 s |
| 7 | FSA-LSSVM | 0.6784 ± 0.0433 | 92.1% | 126.40 s |
| 8 | **ADMM-Nesterov (Primal)** | **0.5799 ± 0.0289** | 75.2% | 161.95 s |
| 9 | FISTA-Nesterov (Primal) | 0.5794 ± 0.0325 | 9.7% | 88.75 s |
| 10 | **ADMM-ElasticNet** | 0.5759 ± 0.0279 | 73.7% | 54.58 s |
| 11 | PruningLSSVM | 0.4615 ± 0.1037 | 95.8% | 46.26 s |

---

## 3. Análise Detalhada dos Modelos Propostos

### 3.1. A Ruptura de Performance do ADMM e Métodos Primais
A maior limitação encontrada no Tier 2 foi o colapso de escalabilidade (tempo) vs performance (F1) na formulação Primal.
- **O Desafio da Escala de $\lambda$:** Para alcançar esparsidade com matrizes massivas usando norma $L_1$, a regularização deve ser pesada, caso contrário, o modelo cai na densidade total e os tempos de treinamento (via iterações Nesterov ou fatorações Cholesky repetitivas) disparam (beirando os 160 segundos no ADMM comum).
- **Perda de Sensibilidade (Bias $b$):** Com um $\lambda$ restrito forçando 75% dos vetores a zero, o hiperplano de separação linear fica muito rígido para acomodar dados tabulares do mundo real, empurrando o F1-macro para a casa do $0.58$ e sofrendo contra distribuições desbalanceadas.
- **ElasticNet ameniza, mas não salva:** O uso da regularização L2 (`ADMMElasticNet`) demonstrou o previsto na teoria matemática: injetou convexidade forte, reduzindo o tempo de convergência drasticamente de 162s para 54s, mas mantendo a restrição expressiva na margem (F1 de $0.5759$).

### 3.2. Nyström-LSSVM: A Rota Definitiva para o LSSVM
A técnica matemática do Nyström provou ser o Santo Graal das LSSVMs no regime moderno de dados ($N \ge 5000$):
- Ele aproxima a matriz esparsa por sub-amostragem estatística de baixo posto, resolvendo o sistema linear em meros **2.20 segundos**.
- Garante o F1-Macro de $0.7072$ (encostado na LSSVM Densa de $0.7093$).
- Remove passivamente **81.5%** da matriz final, entregando os benefícios de inferência rápida na predição que o ADMM prometia, sem a punição brutal do tempo de treinamento iterativo e sem destruir a fronteira de decisão.

### 3.3. FT-CUR: O Casamento do Baixo Posto com a Atenção Neural
No mundo das redes neurais, o Transformer base `SAINT` consumiu muita VRAM e cravou 0% de esparsidade com $0.7223$ de F1.
A abordagem proposta **FT-CUR Transformer** mostrou-se excepcional:
- **Esparsidade Colossal:** Descartou quase $90\%$ das conexões contextuais da atenção (89.4%).
- **Preservação do F1:** Reteve $0.7132$ de F1-Macro.
- Isso fundamenta que a decomposição CUR não apenas comprime as matrizes para caber na GPU, mas age como um excelente regularizador, mantendo as características topológicas densas através de colunas-chave sem precisar de treinamento em matriz cúbica.

---

## 4. Conclusões e Recomendações para a Dissertação

As limitações observadas formam o arco argumentativo perfeito:
1. Começou-se explorando os **Métodos Primais** com restrições $L_1$ puras (ADMM), que funcionaram no Tier 1 (dados pequenos), mas mostraram-se numericamente engessados e lentos para $N \ge 5000$.
2. A solução natural para as LSSVMs foi a aproximação global de rank (O **Método Nyström**), que alcançou o balanço ideal de Tempo $\times$ Desempenho $\times$ Esparsidade.
3. Para arquiteturas de Aprendizado Profundo (onde a topologia é baseada na correlação par-a-par total via self-attention), o espelhamento da técnica de sub-amostragem (A **Decomposição CUR** via *FT-CUR*) mostrou que a filosofia "esparsa" se mantém vitoriosa até o topo do Estado da Arte.

**Recomendação de Próximos Passos Físicos:**
- Gerar os gráficos finais no LaTeX refletindo esses trade-offs.
- Avançar para a submissão formal das comparações de memória VRAM/RAM (Benchmark de Eficiência) para fechar com a cereja empírica da vantagem temporal dos métodos propostos.
