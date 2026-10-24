#!/bin/zsh
# Why the dry-fire penalty hurt the aim: aim signal (aim_consistency) and trunk drift (aim_probe replays) at 2M,
# experiment vs control, seeds 1 and 3. No training.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-ab/aimcheck
SRV=../teeworlds-0.7.5/build-v4
one() {  # name run-dir penalty
  .venv/bin/python bench/aim_consistency.py --server-dir $SRV --run-dir $2 --checkpoint ckpt_0489.pt --batches 16 --dry-fire-penalty $3 --out $D/cons_$1.json > $D/cons_$1.log 2>&1
  .venv/bin/python bench/aim_probe.py --server-dir $SRV --run-dir $2 --checkpoints ckpt_0489.pt --batches 3 --kl-batches 3 --dry-fire-penalty $3 --out $D/probe_$1.json > $D/probe_$1.log 2>&1
}
one ctl_s1 runs/phase3-ab-mask/seed1 0 &
sleep 60; one exp_s1 runs/phase3-fire-ab/short-s1-dry0.0005 0.0005 &
wait
one ctl_s3 runs/phase3-ab-mask/seed3 0 &
sleep 60; one exp_s3 runs/phase3-fire-ab/short-s3-dry0.0005 0.0005 &
wait
echo done > $D/status
