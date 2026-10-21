# Learnability v4: aim_target_kl 0.015, 5M (2026-10-21)

5M learner steps against the frozen ckpt_0000, seed 1; protocol and choice of setting in
`../phase3-aim-stabilisation/README.md`. Result **0/4** at 5M: ratio 0.88, reward -0.003, aim cosine
-0.46, trend tau -0.14. The best point was 2M (`ckpt_0489`: ratio 1.99, hit 0.0244, aim cosine +0.34, but
0 wins and 0 losses). Diagnostics (`diagnose.json`, `credit_audit_*.json`, `aim_probe.txt`) traced the
late decline to the raw aim mean growing without bound; the follow-up is in `../phase3-learnability-v5-bounded`.
