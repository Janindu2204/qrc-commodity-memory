#!/bin/bash
# Launcher for long_prereg_dom4.json: rolling blocks -> rolling report -> held-out blocks (one shot) -> held-out report. Memory-aware.
cd "$(dirname "$0")"
sha256sum -c long_prereg_dom4.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p long_preds_dom4_logs
runq() {  # $1 = job file
  while read -r script a b; do
    while true; do
      n=$(pgrep -fc "python dom4_(run|daily).py"); av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      if [ "$n" -lt 8 ] && [ "$av" -gt 1800 ]; then break; fi; sleep 20
    done
    ( ../../.venv/bin/python $script $a $b > long_preds_dom4_logs/${script%.py}_${a}_${b}.log 2>&1 || echo "FAILED $script $a $b" ) &
    sleep 15
  done < "$1"
  wait
}
echo "start rolling $(date)"; runq dom4_jobs_rolling.txt; echo "rolling blocks done $(date)"
../../.venv/bin/python dom4_report.py rolling > dom4_report_rolling.txt 2>&1; echo "rolling report exit $? $(date)"
echo "start heldout $(date)"; runq dom4_jobs_heldout.txt; echo "heldout blocks done $(date)"
../../.venv/bin/python dom4_report.py heldout > dom4_report_heldout.txt 2>&1; echo "heldout report exit $? $(date)"
