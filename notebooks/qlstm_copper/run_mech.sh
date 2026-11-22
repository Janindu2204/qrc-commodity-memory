#!/bin/bash
cd "$(dirname "$0")"; sha256sum -c long_prereg_mech.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p mech_logs; echo "start $(date)"
for f in DOM DOMG1 ISING ESN SAS ESNB SASB; do ( ../../.venv/bin/python mech_run.py $f > mech_logs/cap_$f.log 2>&1 || echo "FAILED $f" ) & done
for g in H M F; do ( ../../.venv/bin/python mech_copper.py $g > mech_logs/copper_$g.log 2>&1 || echo "FAILED copper $g" ) & done
wait; echo "jobs done $(date)"; ../../.venv/bin/python mech_report.py > mech_report.txt 2>&1; echo "report exit $? $(date)"
