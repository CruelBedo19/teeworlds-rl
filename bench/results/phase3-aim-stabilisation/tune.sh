#!/bin/bash
# Short aim-stabilisation trials: 1.5M learner steps each vs frozen ckpt_0000, same protocol as v3.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
COMMON="--server-dir ../teeworlds-0.7.5/build-release --envs 32 --updates 367 --max-minutes 40 --checkpoint-steps 500000 --aim gaussian --seed 1 --threads 4 --compile --opponent initial"
run() { name=$1; shift; timeout 50m .venv/bin/python scripts/train_ppo.py $COMMON --run-dir runs/overnight/tune-$name "$@" > runs/overnight/tune-$name.log 2>&1; echo "$name exit $?" >> runs/overnight/tune.status; }
run kl015 --aim-target-kl 0.015 &
sleep 20
run lr03 --aim-lr-scale 0.3 &
sleep 20
run lr05kl02 --aim-lr-scale 0.5 --aim-target-kl 0.02 &
wait
echo done >> runs/overnight/tune.status
