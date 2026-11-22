#!/bin/bash
cd "$(dirname "$0")"; sha256sum -c long_prereg_budget.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
until grep -q "report exit" run_rmap2.out; do sleep 30; done      # wait for the regime map (memory)
echo "start $(date)"; ../../.venv/bin/python budget_run.py || echo "FAILED budget_run"
../../.venv/bin/python budget_report.py > budget_report.txt 2>&1; echo "report exit $? $(date)"
