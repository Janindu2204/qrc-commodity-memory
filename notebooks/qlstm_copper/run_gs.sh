#!/bin/bash
# Gold/silver registration (PROGRESS/34). Order: hash check -> build grids -> all blocks (both assets) -> mechanism probe -> report. Memory-aware.
cd "$(dirname "$0")"; sha256sum -c gs_registration.sha256 || { echo "HASH MISMATCH"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; mkdir -p gs_logs; echo "start $(date)"
for S in GC SI; do ../../.venv/bin/python gs_hf_build.py $S > gs_logs/build_$S.log 2>&1 && ../../.venv/bin/python gs_hf2_build.py $S > gs_logs/build2_$S.log 2>&1 || { echo "FAILED build $S"; exit 1; }; done
while read -r S script g y; do
  while [ "$(pgrep -fc 'python gs_(dom|dom2|dom3|dom4|lin)_run.py')" -ge 10 ] || [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -lt 1800 ]; do sleep 15; done
  ( GS_ASSET=$S ../../.venv/bin/python $script $g $y > gs_logs/${S}_${script%.py}_${g}_${y}.log 2>&1 || echo "FAILED $S $script $g $y" ) &
  sleep 5
done < gs_jobs.txt
wait; echo "blocks done $(date)"
for S in GC SI; do for g in H M F; do GS_ASSET=$S ../../.venv/bin/python gs_mech.py $g > gs_logs/mech_${S}_$g.log 2>&1 || echo "FAILED mech $S $g"; done; done
../../.venv/bin/python gs_report.py > gs_report.txt 2>&1; echo "report exit $? $(date)"
