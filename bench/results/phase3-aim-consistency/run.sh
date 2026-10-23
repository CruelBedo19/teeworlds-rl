#!/bin/zsh
# aim gradient consistency, 32 batches per checkpoint, no training
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-3seed/consistency
run() { .venv/bin/python bench/aim_consistency.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/phase3-3seed/$1 --checkpoint $2 --batches 32 --out $D/$1_$2.json > $D/$1_$2.log 2>&1 }
run seed1 ckpt_1099.pt &
sleep 60; run seed3 ckpt_0611.pt &
sleep 60; run seed3 ckpt_0794.pt &
wait
echo done > $D/status
