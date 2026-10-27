#!/bin/bash
# Collapse probe over the drift, seed 1 then seed 3, one at a time (bench/collapse_probe.py, 4 batches each).
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
C=runs/phase3-long/collapse
S=../teeworlds-0.7.5/build-v4
.venv/bin/python bench/collapse_probe.py --server-dir $S --run-dir runs/phase3-long/seed1 --checkpoints ckpt_3663.pt ckpt_4883.pt ckpt_5127.pt ckpt_5372.pt ckpt_5616.pt ckpt_5860.pt ckpt_6104.pt ckpt_7325.pt --batches 4 --out $C/seed1.json > $C/seed1.log 2>&1
.venv/bin/python bench/collapse_probe.py --server-dir $S --run-dir runs/phase3-long/seed3 --checkpoints ckpt_3663.pt ckpt_4883.pt ckpt_5372.pt ckpt_5616.pt ckpt_5860.pt ckpt_6104.pt ckpt_6348.pt ckpt_7325.pt --batches 4 --out $C/seed3.json > $C/seed3.log 2>&1
echo done > $C/status
