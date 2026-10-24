#!/bin/zsh
# Phase E: the 1M result was unclear (fire better in all seeds, aim lower), so each short run continues to
# 2M (489 updates) from its last checkpoint, then bench/diagnose.py on the 2M checkpoint. Control:
# runs/phase3-ab-mask/seed*/ckpt_0489.pt.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-ab
for seed in "$@"; do
  R=$D/short-s$seed-dry0.0005
  echo "$(date '+%F %T') extend start $R" >> $D/status
  .venv/bin/python scripts/train_ppo.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir $R --resume $R/ckpt_final.pt --envs 32 --updates 489 --max-minutes 120 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --seed $seed --dry-fire-penalty 0.0005 > $R.extend.log 2>&1
  echo "$(date '+%F %T') extended $R exit $?" >> $D/status
  .venv/bin/python bench/diagnose.py --server-dir ../teeworlds-0.7.5/build-v4 --opponent $R/ckpt_0000.pt --checkpoints $R/ckpt_0489.pt --out $R.diag2m.json > $R.diag2m.log 2>&1
  echo "$(date '+%F %T') diagnosed 2M $R exit $?" >> $D/status
done
echo "$(date '+%F %T') extend done $*" >> $D/status
