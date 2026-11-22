#!/bin/bash
# Launcher for long_prereg_rmap.json: all synthetic datasets -> report. Memory-aware.
cd "$(dirname "$0")"
sha256sum -c long_prereg_rmap.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p rmap_logs
echo "start $(date)"
while read -r m st sh r; do
  while true; do
    n=$(pgrep -fc "python rmap_run.py"); av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$n" -lt 10 ] && [ "$av" -gt 1500 ]; then break; fi; sleep 10
  done
  ( ../../.venv/bin/python rmap_run.py $m $st $sh $r > rmap_logs/${m}_${st}_${sh}_${r}.log 2>&1 || echo "FAILED $m $st $sh $r" ) &
  sleep 2
done < rmap_jobs.txt
wait; echo "datasets done $(date)"
../../.venv/bin/python rmap_report.py > rmap_report.txt 2>&1; echo "report exit $? $(date)"
