#!/bin/bash
cd "$(dirname "$0")"; sha256sum -c long_prereg_lin.sha256 && sha256sum -c long_prereg_budget2.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; mkdir -p lin_logs; echo "start $(date)"
( ../../.venv/bin/python budget2_run.py > budget2_report.txt 2>&1; echo "budget2 exit $? $(date)" ) &
for g in H M F; do for y in $(seq 2013 2026); do
  while [ "$(pgrep -fc 'python lin_run.py')" -ge 4 ] || [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -lt 2000 ]; do sleep 5; done
  ( ../../.venv/bin/python lin_run.py $g $y > lin_logs/${g}_${y}.log 2>&1 || echo "FAILED lin $g $y" ) &
  sleep 1
done; done
wait; echo "jobs done $(date)"; ../../.venv/bin/python lin_report.py > lin_report.txt 2>&1; echo "lin report exit $? $(date)"
