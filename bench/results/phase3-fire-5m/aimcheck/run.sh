#!/bin/bash
# Aim push signal at 5M, both arms, training-free (adv_decompose.py, 16 batches each), one at a time.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
S=../teeworlds-0.7.5/build-v4
O=runs/phase3-fire-5m/aimcheck
for s in 1 2 3; do
  .venv/bin/python bench/adv_decompose.py --server-dir $S --run-dir runs/phase3-fire-5m/seed$s --checkpoint ckpt_1221.pt --dry-fire-penalty 0.0005 --batches 16 --out $O/pen_s$s.json > $O/pen_s$s.log 2>&1
  .venv/bin/python bench/adv_decompose.py --server-dir $S --run-dir runs/phase3-ab-mask/seed$s --checkpoint ckpt_1221.pt --dry-fire-penalty 0 --batches 16 --out $O/ctl_s$s.json > $O/ctl_s$s.log 2>&1
done
echo done > $O/status
