#!/bin/zsh
# Phase A/B fire credit audit on the shots-only runs: ckpt_0000, 1M, 3M, 5M, 24 batches each, no training
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-audit
run() { .venv/bin/python bench/fire_audit.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir runs/phase3-ab-mask/seed$1 --checkpoints ckpt_0000.pt ckpt_0245.pt ckpt_0733.pt ckpt_1221.pt --batches 24 --out $D/audit_seed$1.json > $D/audit_seed$1.log 2>&1 }
run 1 &
sleep 60; run 2 &
sleep 60; run 3 &
wait
echo done > $D/status
