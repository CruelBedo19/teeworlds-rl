#!/bin/zsh
# Aim question of phase3-fire-5m: seeds 4 and 5, both arms, continue 2M -> 5M (1221 updates) from ckpt_final,
# one at a time; then diagnose (64 matches vs the run's ckpt_0000) at 3, 4 and 5M.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-ab
for seed in 4 5; do
  for c in 0 0.0005; do
    R=$D/rep-s$seed-dry$c
    echo "$(date '+%F %T') extend5m start $R" >> $D/status
    timeout 150m .venv/bin/python scripts/train_ppo.py --server-dir ../teeworlds-0.7.5/build-v4 --run-dir $R --resume $R/ckpt_final.pt --envs 32 --updates 1221 --max-minutes 140 --checkpoint-steps 250000 --aim gaussian --threads 4 --compile --opponent initial --aim-target-kl 0.015 --aim-mean bounded --aim-mask shots --env-timeout 600 --seed $seed --dry-fire-penalty $c > $R.extend5m.log 2>&1
    echo "$(date '+%F %T') extend5m trained $R exit $?" >> $D/status
  done
done
for seed in 4 5; do
  for c in 0 0.0005; do
    R=$D/rep-s$seed-dry$c
    timeout 120m .venv/bin/python bench/diagnose.py --server-dir ../teeworlds-0.7.5/build-v4 --opponent $R/ckpt_0000.pt --checkpoints $R/ckpt_0733.pt $R/ckpt_0977.pt $R/ckpt_1221.pt --out $R.diag5m.json > $R.diag5m.log 2>&1 &
  done
  wait
done
echo "$(date '+%F %T') extend5m done" >> $D/status
