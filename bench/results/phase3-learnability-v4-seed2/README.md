# Learnability v4, seed 2 (2026-10-21)

The v4 configuration (raw aim mean, aim_target_kl 0.015) with seed 2, run next to v5. Tables and
reading are in `../phase3-learnability-v5-bounded/README.md`. Result: 3/4 at 5M (fails criterion 2, no
wins). The 2M peak of seed 1 does not repeat: seed 2's ratio rises later (1.52 at 5M) and its aim cosine
stays negative at most points.
