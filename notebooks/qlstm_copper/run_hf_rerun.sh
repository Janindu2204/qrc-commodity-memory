#!/bin/bash
cd "$(dirname "$0")"
sha256sum -c long_prereg_hf.sha256 --quiet 2>&1 | grep -v "run_hf.sh\|hf_jobs.txt" ; echo "start $(date)"
export OMP_NUM_THREADS=2
cat hf_jobs_rerun.txt | xargs -P 5 -L 1 bash -c '../../.venv/bin/python hf_run.py $0 $1 > long_preds_hf_logs/$0_$1.log 2>&1 || echo "FAILED $0 $1"'
echo "blocks done $(date)"
../../.venv/bin/python hf_report.py > hf_report.txt 2>&1; echo "report exit $? $(date)"
