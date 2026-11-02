V7 STRICT RRT-SAMPLE GAN - START HERE

Package: V7_7_GENERALIZATION_GAN_FINAL

Method: basic RRT with a sample-conditioned learned local steering function.

Required operation order:
valid random/goal-biased sample -> Euclidean-nearest tree parent -> GAN(parent,sample,obstacles) -> safety checks -> add at most one child.

Goal policy: the search succeeds when that one child enters the goal region. There is no hidden multi-node goal-connection chain.
Benchmark policy: invalid starts, invalid goals, and cases already inside the goal radius never enter solve-rate statistics.

Condition: [parent_x,parent_y,parent_vx,parent_vy,sample_x,sample_y,absolute_obstacle_x_y_r...]
Generator output: [delta_x,delta_y,next_vx,next_vy]
State reconstruction: [parent_x+delta_x,parent_y+delta_y,next_vx,next_vy].
Training target sources: ['recorded_rrt_sample_xy', 'recorded_rrt_sample_xy', 'recorded_rrt_sample_xy']

All three sources must say recorded_rrt_sample_xy. V7 refuses to run otherwise.

Dataset rows:
training: 48000
validation: 6000
testing: 6000

Scientific boundaries:
- validation/testing never receive gradient updates
- testing is excluded from checkpoint selection
- best-of is an oracle diagnostic, not deployment selection
- no fake 98% claim
- no MPC or state-refiner claim
- this is basic RRT, not RRT*
- reported path cost is Euclidean path length, not the paper's LQR objective
- benchmark uncertainty primarily uses a 2,000-replicate environment-cluster bootstrap; the unclustered row-level Wilson interval is retained only as a diagnostic
- exactly 100 valid nontrivial cases are evaluated per difficulty and split
- unless explicitly skipped, held-out testing rows are rerun with a paired classical direct-steering basic-RRT baseline
