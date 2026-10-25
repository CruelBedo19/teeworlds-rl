#!/bin/zsh
# Re-run the two 5M diagnoses that failed on a port clash (infrastructure), one at a time.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
D=runs/phase3-fire-ab
for R in $D/rep-s4-dry0 $D/rep-s5-dry0.0005; do
  mv $R.diag5m.log $R.diag5m.portclash.log
  timeout 120m .venv/bin/python bench/diagnose.py --server-dir ../teeworlds-0.7.5/build-v4 --opponent $R/ckpt_0000.pt --checkpoints $R/ckpt_0733.pt $R/ckpt_0977.pt $R/ckpt_1221.pt --out $R.diag5m.json > $R.diag5m.log 2>&1
  echo "$(date '+%F %T') rediag5m $R exit $?" >> $D/status
done
echo "$(date '+%F %T') rediag5m done" >> $D/status
