import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def main():
    # Ordem de leitura: arquivos mais antigos primeiro; os mais novos
    # sobrescrevem por (dataset, model, seed) na deduplicação.
    # - parallel.json (2026-05-31): runs LSSVM/FT (versão pré-correção do
    #   Pruning/ADMM/FISTA + SAINT com val_acc) e exclusivos: DualFISTA,
    #   NystromLSSVMColnorm, OppositeMapsLSSVM, SAINTColnorm.
    # - gpu.json (2026-05-29): FT-Transformer baselines + FT-CUR (val_acc).
    # - ftcur_saint_valloss.json (2026-05-31): FT-CUR/SAINT corrigidos
    #   com early_stop_metric=val_loss — devem sobrescrever os de gpu/parallel.
    # - cpu.json (2026-06-03): Pruning/ADMM/FISTA/Standard/XGBoost reexecutados
    #   após Fix b83243e (Pruning early stopping + Primal lambda scaling) ---
    #   precisa vir por último para sobrescrever parallel.json.
    files = [
        'results/tier2_n5000_parallel.json',
        'results/tier2_n5000_gpu.json',
        'results/tier2_n5000_ftcur_saint_valloss.json',
        'results/tier2_n5000_cpu.json',
    ]

    data = []
    for f in files:
        path = Path(f)
        if path.exists():
            print(f"Lendo {f}...")
            try:
                with open(path) as fp:
                    d = json.load(fp)
                    if isinstance(d, list):
                        data.extend(d)
            except Exception as e:
                print(f"Erro lendo {f}: {e}")

    # Deduplicate
    unique_data = {}
    for r in data:
        if 'model' not in r or 'dataset' not in r or 'seed' not in r:
            continue
        model = r.get('model_variant') or r['model']
        dataset = r['dataset']
        seed = r['seed']
        
        # Override FT-CUR/SAINT with the val_loss ones (from newer file)
        # Assuming the order of files processes the valloss ones last and overwrites
        unique_data[(dataset, model, seed)] = r

    combined = list(unique_data.values())
    
    out_path = Path('results/tier2_n5000_combined.json')
    with open(out_path, 'w') as fp:
        json.dump(combined, fp, indent=2)
    print(f"\nSalvo {len(combined)} execuções consolidadas em {out_path}")

    # Generate summary stats
    model_stats = defaultdict(lambda: {'f1': [], 'sparsity': [], 'time': []})
    for r in combined:
        model = r.get('model_variant') or r['model']
        f1 = r.get('f1_macro', 0)
        
        n_sv = r.get('n_support_vectors', None)
        n_total = r.get('n_train', 5000)
        sparsity = 0
        if n_sv is not None:
            sparsity = (1.0 - (n_sv / n_total)) * 100
            
        t = r.get('train_time_s', 0)
        
        model_stats[model]['f1'].append(f1)
        model_stats[model]['sparsity'].append(sparsity)
        model_stats[model]['time'].append(t)

    print(f"\n{'Model':<25} | {'F1-Macro':<15} | {'Sparsity %':<15} | {'Fit Time (s)':<15}")
    for m, stats in sorted(model_stats.items(), key=lambda x: np.mean(x[1]['f1']), reverse=True):
        f1_mean = np.mean(stats['f1'])
        f1_std = np.std(stats['f1'])
        sp_mean = np.mean(stats['sparsity'])
        t_mean = np.mean(stats['time'])
        print(f"{m:<25} | {f1_mean:.4f}±{f1_std:.4f} | {sp_mean:.1f}% | {t_mean:.2f}s")

if __name__ == '__main__':
    main()
