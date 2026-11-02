V7 STRICT RRT-SAMPLE GAN BENCHMARK EXPLANATION

Solve rate is the percentage of valid, nontrivial cases that reached the goal region without crossing an obstacle.
Invalid starts, invalid goals, and cases already within the goal radius are excluded before difficulty bucketing and are reported in benchmark_eligibility_diagnostics.json.
Each solve-rate value includes an environment-cluster bootstrap 95% interval as the primary uncertainty estimate. A row-level Wilson interval is retained only as a transparent unclustered diagnostic.
Solved-path length is accumulated Euclidean distance, not the original LQR objective. It is computed only for solved cases; mixing failed partial paths into this average would be misleading.
Runtime is planner computation time across all attempts for a case; image rendering time is excluded.
Tree/path counters describe the selected attempt; separate total_* columns record effort accumulated across every executed attempt.
Tree nodes include the root plus accepted branches, while rejected collision/boundary proposals are tracked separately in the case CSV.
Runtime contracts verify one free-space sample, one Euclidean-nearest parent, the same GAN target, at most one accepted child, and zero accepted obstacle or configuration-space-boundary violations per iteration.
GAN and fallback acceptances are counted separately. Fallback steering is eligible only when no GAN proposal survives all steering, safety, progress, and duplicate checks.
Final goal distance is especially useful for failed cases because it shows whether failure occurred near the goal or in a true dead end.
Easy, medium, and hard are relative thirds of each split according to start-goal distance, direct-path clearance, and obstacle count.
The ranking score is distance + 4*max(0, 0.25 - direct_path_clearance) + 0.06*obstacle_count; it ranks cases but is not a physical cost.
The held-out classical comparison uses identical testing rows and seeds, but replaces GAN/discriminator steering with one direct bounded extension toward each sample.

Exact counts and values are in 02_OVERALL_BENCHMARK_DATA; the PNG files are visual summaries of those same values.
