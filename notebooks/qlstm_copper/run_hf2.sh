#!/bin/bash
# Launcher for long_prereg_hf2.json: memory-aware (another process on this machine uses ~3.7 GB). Rolling blocks -> rolling report -> 2026 one-shot -> held-out report.
cd "$(dirname "$0")"
sha256sum -c long_prereg_hf2.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=2; mkdir -p long_preds_hf2_logs
runq() {  # $1 = job file
  while read -r mode g y; do
    while true; do
      n=$(pgrep -fc "python hf2_run.py"); av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      if [ "$n" -lt 5 ] && [ "$av" -gt 2300 ]; then break; fi; sleep 20
    done
    ( ../../.venv/bin/python hf2_run.py $mode $g $y > long_preds_hf2_logs/${mode}_${g}_${y}.log 2>&1 || echo "FAILED $mode $g $y" ) &
    sleep 40
  done < "$1"
  wait
}
echo "start rolling $(date)"; runq hf2_jobs_rolling.txt; echo "rolling blocks done $(date)"
../../.venv/bin/python hf2_report.py rolling > hf2_report_rolling.txt 2>&1; echo "rolling report exit $? $(date)"
echo "start heldout $(date)"; runq hf2_jobs_heldout.txt; echo "heldout blocks done $(date)"
../../.venv/bin/python hf2_report.py heldout > hf2_report_heldout.txt 2>&1; echo "heldout report exit $? $(date)"
