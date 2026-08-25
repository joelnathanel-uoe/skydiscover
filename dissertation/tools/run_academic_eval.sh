#!/bin/bash
cd /root/skydiscover || exit 1
for T in A B; do
  D=$(ls -d outputs/adaevolve/fjsp_twt_p1${T}_0728b/checkpoints/checkpoint_* | sed 's/.*_//' | sort -n | tail -1)
  P=outputs/adaevolve/fjsp_twt_p1${T}_0728b/checkpoints/checkpoint_${D}/best_program.py
  echo "=============== program $T (checkpoint_$D) ==============="
  uv run python benchmarks/math/fjsp_twt_insertion/evaluate_academic.py "$P" --verbose 2>&1 \
    | tee /root/skydiscover/academic_eval_p1${T}.log
done
echo "ALL DONE"
