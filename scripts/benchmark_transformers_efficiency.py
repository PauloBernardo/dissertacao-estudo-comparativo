#!/usr/bin/env python3
"""
Benchmark computacional para mensurar uso bruto de memória RAM (Sistema) e VRAM (GPU)
além do tempo (s) entre os modelos baseados em Transformers.
"""
import sys
import time
import numpy as np
import torch
import gc
import json
import threading
from pathlib import Path

# Adicionar root ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ft_transformer_saint_wrapper import SAINTColnorm
from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
from src.models.transformers.ft_transformer import FTTransformer

class RamMonitor:
    """Monitora o pico de RAM (VmRSS) via /proc/self/status em uma thread separada."""
    def __init__(self):
        self.keep_measuring = True
        self.peak_ram_kb = 0
        self.start_ram_kb = 0
        self._record_start()

    def _read_rss(self):
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1])
        except Exception:
            return 0
        return 0

    def _record_start(self):
        self.start_ram_kb = self._read_rss()
        self.peak_ram_kb = self.start_ram_kb

    def measure(self):
        while self.keep_measuring:
            current_ram = self._read_rss()
            if current_ram > self.peak_ram_kb:
                self.peak_ram_kb = current_ram
            time.sleep(0.01)

def get_best_params(variant, json_file, key_format):
    path = Path(json_file)
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    key = key_format.format(variant=variant)
    if key in data:
        return data[key].get("best_params", {})
    return {}

def measure_memory_and_time(model_factory, X, y):
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Inicia monitoramento de RAM
    monitor = RamMonitor()
    thread = threading.Thread(target=monitor.measure)
    thread.start()
    
    model = model_factory()
    
    t0 = time.perf_counter()
    success = True
    try:
        model.fit(X, y)
        t1 = time.perf_counter()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cublas" in str(e).lower():
            t1 = time.perf_counter()
            success = False
        else:
            raise e
            
    # Para o monitoramento de RAM
    monitor.keep_measuring = False
    thread.join()
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) # in MB
    else:
        peak_vram = 0.0 # CPU mode
        
    # Pico real usado (Pico Total - Inicial)
    peak_ram = max(0, monitor.peak_ram_kb - monitor.start_ram_kb) / 1024 # in MB
        
    return (t1 - t0), peak_ram, peak_vram, success

def run_benchmark():
    if not torch.cuda.is_available():
        print("Aviso: CUDA não disponível. O benchmark de VRAM registrará 0MB.")

    sizes = [1000, 3000, 5000, 10000]
    D = 14 # Features do ADULT
    
    # Carregar os parâmetros otimizados para o dataset ADULT
    gpu_tuning = "results/tuning/best_params_tier2_n5000_gpu.json"
    valloss_tuning = "results/tuning/best_params_ftcur_saint_valloss.json"
    
    params_softmax = get_best_params("FTTransformer_softmax", gpu_tuning, "{variant}__ADULT")
    params_topk = get_best_params("FTTransformer_topk", gpu_tuning, "{variant}__ADULT")
    params_entmax = get_best_params("FTTransformer_entmax", gpu_tuning, "{variant}__ADULT")
    params_sparsemax = get_best_params("FTTransformer_sparsemax", gpu_tuning, "{variant}__ADULT")
    
    params_cur = get_best_params("FTTransformerCURColnorm", valloss_tuning, "{variant}__ADULT__val_loss")
    params_saint = get_best_params("SAINTColnorm", valloss_tuning, "{variant}__ADULT__val_loss")
    
    # Forçar apenas 1 epoch para o benchmark de eficiência
    for p in [params_softmax, params_topk, params_entmax, params_sparsemax]:
        p["max_epochs"] = 1
    for p in [params_cur, params_saint]:
        p["epochs"] = 1
        p["early_stop_metric"] = "val_loss"
    
    # Default para FT baselines caso o json não tenha a chave de attention_type salva
    params_topk.setdefault("attention_type", "topk")
    params_topk.setdefault("topk_ratio", 0.10)
    params_softmax.setdefault("attention_type", "softmax")
    params_entmax.setdefault("attention_type", "entmax")
    params_entmax.setdefault("alpha", 1.5)
    params_sparsemax.setdefault("attention_type", "sparsemax")

    models = [
        ("FT-Softmax", lambda: FTTransformer(**params_softmax)),
        ("FT-TopK", lambda: FTTransformer(**params_topk)),
        ("FT-Entmax", lambda: FTTransformer(**params_entmax)),
        ("FT-Sparsemax", lambda: FTTransformer(**params_sparsemax)),
        ("FT-CUR", lambda: FTTransformerCURColnorm(**params_cur)),
        ("SAINT", lambda: SAINTColnorm(**params_saint))
    ]
    
    print(f"{'Modelo':<15} | {'N':<6} | {'Tempo (s)':<10} | {'RAM CPU (MB)':<15} | {'VRAM GPU (MB)':<15}")
    print("-" * 73)
    
    for name, factory in models:
        for n in sizes:
            np.random.seed(42)
            X = np.random.randn(n, D).astype(np.float32)
            y = np.random.randint(0, 2, size=n).astype(np.int64)
            
            t, ram_cpu, vram_gpu, success = measure_memory_and_time(factory, X, y)
            if success:
                print(f"{name:<15} | {n:<6} | {t:<10.2f} | {ram_cpu:<15.1f} | {vram_gpu:<15.1f}")
            else:
                print(f"{name:<15} | {n:<6} | {'OOM':<10} | {'OOM':<15} | {'OOM':<15}")
                break # Pula para o próximo modelo se estourar memória

if __name__ == '__main__':
    run_benchmark()
