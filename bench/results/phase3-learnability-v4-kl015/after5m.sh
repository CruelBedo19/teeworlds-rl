#!/bin/bash
# Decision tree after the 5M run, unattended. Writes progress to runs/overnight/after5m.status.
cd "${REPO_DIR:?set REPO_DIR to the repo checkout}"
S=runs/overnight/after5m.status
R=runs/overnight/learn-5m-kl015
SRV=../teeworlds-0.7.5/build-release
PY=.venv/bin/python
log() { echo "$(date '+%F %T') $*" >> $S; }
until grep -q "pipeline exit" runs/overnight/regress/status 2>/dev/null; do sleep 60; done
log "5M pipeline finished: $(grep 'pipeline exit' runs/overnight/regress/status)"
# the standalone baseline evaluation (the pipeline's own one raced the trainer's start)
until [ -f $R/eval/ckpt_0000.json ] || ! pgrep -f "eval_suite.py.*learn-5m-kl015 --checkpoint ckpt_0000.pt" >/dev/null; do sleep 30; done
[ -f $R/eval/ckpt_0000.json ] || timeout 30m $PY scripts/eval_suite.py --server-dir $SRV --run-dir $R --checkpoint ckpt_0000.pt >> runs/overnight/baseline_eval.log 2>&1
$PY bench/learn_pipeline.py --summarize --run-dir $R --eval-at 500000,1000000,1500000,2000000,3000000,4000000,5000000 > $R/summarize.log 2>&1
PASS=$($PY -c "import json; print(int(json.load(open('$R/summary.json')).get('criteria_at_last_point', {}).get('all_four', False)))")
log "criteria all_four=$PASS"
ck() { $PY -c "import json,os; print(os.path.join('$R', json.load(open('$R/eval/at_$1.json'))['current'].split('/')[-1]))"; }
if [ "$PASS" = "1" ]; then
  FINAL=$(ck 5000000)
  log "PASS branch: strong evaluation of $FINAL"
  for opp in $R/ckpt_0000.pt random $(ck 1000000) $(ck 3000000); do
    n=$(basename $opp .pt)
    timeout 90m $PY bench/strong_eval.py --server-dir $SRV --checkpoint $FINAL --opponents $opp --seeds 8 --out runs/overnight/strong_$n.json > runs/overnight/strong_$n.log 2>&1 &
    sleep 20
  done
  wait
  # pre-registered strong pass: vs ckpt_0000 over 512 episodes ratio >= 1.25, wins > losses,
  # reward mean - 2 SE > 0 over the 8 seeds, and at least 6 of 8 seeds with positive reward
  STRONG=$($PY -c "
import json
r = json.load(open('runs/overnight/strong_ckpt_0000.json'))['opponents']
o = next(iter(r.values())); p, s = o['pooled'], o['over_seeds']
print(int(p['damage_ratio'] >= 1.25 and p['wins'] > p['losses'] and s['reward_mean'] - 2 * s['reward_se'] > 0 and s['seeds_with_positive_reward'] >= 6))")
  log "strong evaluation pass=$STRONG"
  if [ "$STRONG" = "1" ]; then
    log "self-play pilot from $FINAL"
    timeout 150m $PY bench/learn_pipeline.py --run-dir runs/overnight/selfplay-pilot --eval-at 1000000,2000000,4000000,6000000,8000000 -- \
      --envs 32 --updates 977 --max-minutes 120 --checkpoint-steps 500000 --aim gaussian --seed 1 --threads 6 --compile \
      --aim-target-kl 0.015 --init-from $FINAL > runs/overnight/selfplay-pilot.out 2>&1
    log "self-play pilot exit $?"
    timeout 60m $PY bench/diagnose.py --server-dir $SRV --opponent $R/ckpt_0000.pt --checkpoints $FINAL $(ls runs/overnight/selfplay-pilot/ckpt_[0-9]*.pt | tail -1) --out runs/overnight/diag_selfplay.json > runs/overnight/diag_selfplay.log 2>&1
    log "self-play diagnostics exit $?"
  fi
else
  log "FAIL branch: diagnostics"
  CKS="$R/ckpt_0000.pt"
  for t in 500000 1000000 2000000 3000000 4000000 5000000; do [ -f $R/eval/at_$t.json ] && CKS="$CKS $(ck $t)"; done
  timeout 90m $PY bench/diagnose.py --server-dir $SRV --opponent $R/ckpt_0000.pt --checkpoints $CKS --out runs/overnight/diag_5m.json > runs/overnight/diag_5m.log 2>&1 &
  sleep 30
  LAST=$(basename $(echo $CKS | tr ' ' '\n' | tail -1))
  timeout 60m $PY bench/credit_audit.py --server-dir $SRV --run-dir $R --learner $LAST --rollouts 40 --envs 32 --traces 4 --out runs/overnight/audit_5m_final.json > runs/overnight/audit_5m_final.log 2>&1 &
  sleep 30
  timeout 60m $PY bench/credit_audit.py --server-dir $SRV --run-dir $R --learner $(basename $(ck 500000)) --rollouts 40 --envs 32 --traces 4 --out runs/overnight/audit_5m_early.json > runs/overnight/audit_5m_early.log 2>&1 &
  wait
  log "diagnostics done"
fi
log "after5m done"
