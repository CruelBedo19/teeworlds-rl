#!/bin/zsh
# Is the aim loss under the dry-fire penalty real? Seeds 4 and 5, control (shots-only) and experiment
# (penalty 5e-4), 2M learner steps (489 updates) each, one at a time; diagnose at 1M and 2M.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-ab
for seed in 4 5; do
  for c in 0 0.0005; do
    R=$D/rep-s$seed-dry$c
    echo "$(date '+%F %T') start $R" >> $D/status
    .venv/bin/python scripts/train_ppo.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir $R --envs 32 --updates 489 --max-minutes 120 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --seed $seed --dry-fire-penalty $c > $R.train.log 2>&1
    echo "$(date '+%F %T') trained $R exit $?" >> $D/status
    .venv/bin/python bench/diagnose.py --server-dir ../teeworlds-0.7.5/build-v4 --opponent $R/ckpt_0000.pt --checkpoints $R/ckpt_0245.pt $R/ckpt_0489.pt --out $R.diag.json > $R.diag.log 2>&1
    echo "$(date '+%F %T') diagnosed $R exit $?" >> $D/status
  done
done
echo "$(date '+%F %T') replicate done" >> $D/status
