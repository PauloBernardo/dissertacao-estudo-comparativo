cd /home/paulo/Documentos/dissertacao-estudo-comparativo/sparse-lssvm-transformers-study
while kill -0 530380 2>/dev/null; do sleep 60; done
cd scratch/lssvm_table18_package && ../../.venv/bin/python run_table18_benchmark.py --output table18_l1rkhs.json --variants ADMMNystromLSSVM,FISTANystromLSSVM --repo-root ../.. > run_l1rkhs.log 2>&1
echo BENCH_FIM
