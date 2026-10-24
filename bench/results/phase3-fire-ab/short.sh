#!/bin/zsh
# Short A/B (Phase D): shots-only baseline plus a dry-fire penalty, seed $1, 1M learner steps (245 updates),
# then bench/diagnose.py on ckpt_0000 and the 1M checkpoint. The control is runs/phase3-ab-mask/seed$1
# (identical start: its first updates are reproduced bit for bit at this commit).
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-ab
seed=$1; shift
for c in "$@"; do
  R=$D/short-s$seed-dry$c
  echo "$(date '+%F %T') start $R" >> $D/status
  .venv/bin/python scripts/train_ppo.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir $R --envs 32 --updates 245 --max-minutes 120 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --seed $seed --dry-fire-penalty $c > $R.train.log 2>&1
  echo "$(date '+%F %T') trained $R exit $?" >> $D/status
  .venv/bin/python bench/diagnose.py --server-dir ../teeworlds-0.7.5/build-v4 --opponent $R/ckpt_0000.pt --checkpoints $R/ckpt_0000.pt $R/ckpt_0245.pt --out $R.diag.json > $R.diag.log 2>&1
  echo "$(date '+%F %T') diagnosed $R exit $?" >> $D/status
done
echo "$(date '+%F %T') short done seed $seed" >> $D/status
