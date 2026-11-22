#!/bin/bash
# Launcher for long_prereg_hf.json. Checks hashes, runs 26 blocks (15-minute grid first: it is the larger), then scores.
cd "$(dirname "$0")"
sha256sum -c long_prereg_hf.sha256 || { echo "HASH MISMATCH, not running"; exit 1; }
export OMP_NUM_THREADS=2
echo "start $(date)"
cat hf_jobs.txt | xargs -P 10 -L 1 bash -c '../../.venv/bin/python hf_run.py $0 $1 > long_preds_hf_logs/$0_$1.log 2>&1 || echo "FAILED $0 $1"'
echo "blocks done $(date)"
../../.venv/bin/python hf_report.py > hf_report.txt 2>&1; echo "report exit $? $(date)"
