CLEAR TITLE: VALIDATION DATASET

Method: V7 Strict Basic RRT with Sample-Conditioned GAN Steering

Folders:
01_METRICS
02_V7_STRICT_RRT_SAMPLE_GAN_PICTURES
03_ROLLOUT_DATA

Condition: [parent_x,parent_y,parent_vx,parent_vy,sample_x,sample_y,absolute_obstacle_x_y_r...]
Generator output: [delta_x,delta_y,next_vx,next_vy]
Full state: [parent_x+delta_x,parent_y+delta_y,next_vx,next_vy].

The free-space RRT sample is selected before the Euclidean-nearest parent and is then passed to the GAN.
At most one accepted child is added per RRT iteration. Entering the goal region ends the search; no multi-node connector is used.
Invalid starts/goals and zero-iteration trivial cases are excluded from solve-rate benchmarks and reported separately.
No MPC, state refiner, RRT* rewiring, or one-step picture folder is claimed.
