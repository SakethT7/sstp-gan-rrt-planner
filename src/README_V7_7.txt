V7.7 GENERALIZATION-FOCUSED RRT-SAMPLE GAN
===========================================

WHY V7.6 DID NOT IMPROVE
------------------------
V7.6 was not undertrained. Its hard-mining reports showed approximately
99.93% strict training velocity accuracy. Training mean velocity error fell
from 0.002856 to 0.000496, yet strict held-out validation velocity accuracy
remained about 77%. The validation-selected V7.6 result occurred at epoch 1.
This is environment overfitting, so more epochs or more hard-example sampling
cannot honestly solve it.

Source review found that the previous learned trunk still received
cond[:, 0:6], including absolute parent and sampled-target coordinates. The
large ordered-obstacle MLP could therefore identify training environments.

WHAT V7.7 CHANGES
-----------------
1. Starts from scratch; V7.5/V7.6 warm starts are rejected.
2. Learned layers receive no raw absolute parent/sample coordinates.
3. Motion is represented in the sampled-target frame.
4. Every obstacle, wall, and acceleration bound becomes a shared LQR/CBF/QP
   constraint token containing its local normal, bound, nominal slack,
   unclipped-objective violation, clipped-baseline feasibility, label-free
   single-constraint correction, and geometry.
5. Permutation-equivariant token interaction models two-constraint active-set
   intersections; attention plus mean/max pooling keeps obstacle order irrelevant.
6. Reflection parity averaging plus the sample frame enforces all square D4
   rotations/reflections for arbitrary model weights.
7. Exact D4 transforms augment training only. Validation/testing are unchanged.
8. The generator is much smaller and uses AdamW weight decay.
9. Hard-example oversampling is forced to zero because V7.6 proved it increased
   memorization without increasing validation accuracy.
10. A fixed training probe reports the train/validation velocity gap at every
    evaluation.
11. The untrained nominal-LQR epoch-0 baseline is a selectable checkpoint, so
    optimization can never discard it unless validation genuinely improves.
12. Testing-label summaries remain locked until training and validation model
    selection are complete.
13. Finalization reports empirical elapsed time and ETA for every benchmark
    split, the paired classical baseline, route-image rendering, chart
    generation, verification, and ZIP creation.
14. The final results gallery creates numbered 300-DPI PNG and vector SVG
    figures. Every plotted figure has a corresponding CSV containing the exact
    values shown; no chart data or route is fabricated.
15. The final package preserves the complete source suite, selected deployable
    model, resumable training checkpoint, final-discriminator checkpoint,
    runtime versions, commands, hashes, and verification reports.

The output remains control-consistent: the network predicts one normalized
velocity increment, then position and next velocity are reconstructed through
the exact verified one-step double-integrator dynamics.

ACCURACY RULE
-------------
The deployable deterministic generator must simultaneously achieve:
- position error <= 0.05 on at least 98% of validation rows;
- velocity error <= 0.05 on at least 98% of validation rows; and
- full-state error <= 0.10 on at least 98% of validation rows.

Best-of-16 is an oracle diagnostic and never selects or certifies the model.
Discriminator accuracy is a GAN-balance diagnostic and is not motion accuracy.
Approximately 50% discriminator accuracy can be expected when real and
generated conditional distributions are difficult to distinguish.

No code can guarantee an unobserved 98% result. V7.7 is designed around the
measured bottleneck and will report the result exactly. It will not fabricate a
target hit, loosen thresholds, train on validation/testing, or silently start
the expensive final benchmark below target.

FILES
-----
- build_v7_7_rrt_sample_gan.py: trainer, validation, final evaluation, planner,
  charts, and packaging.
- generate_v7_rrt_sample_dataset.py: unchanged strict V7.3 dataset generator.
- v7_7.sh: protected launcher with consistent arguments.
- monitor_v7_7_training.py: selected/latest validation status.
- render_v7_7_path_pictures.py: optional real rollout picture renderer, run
  automatically during the complete final pipeline.
- build_v7_7_results_gallery.py: creates the numbered real-data chart gallery,
  exact figure CSVs, figure captions, integrity manifest, and final ZIP.
- run_v7_7_final_pipeline.sh: runs held-out evaluation, route rendering, gallery
  construction, verification, and archiving as one protected workflow.
- verify_v7_7_final_package.py: independently rechecks benchmark counts,
  collision/RRT contracts, all disk hashes, every ZIP member, all figure/data
  pairs, all path-image records, source preservation, and checkpoint presence.

NORMAL WORKFLOW
---------------
Place this entire directory at:
  $HOME/V7_7_GENERALIZATION_SOURCE

Then run each command separately:

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" preflight

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" train

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" status

To watch live output:

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" watch

Ctrl+C while watching stops only the log viewer; background training continues.

If WSL or the computer interrupted training:

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" resume

Only after status reports deployable_validation_target_reached: True:

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" finalize

Monitor the complete final pipeline:

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" final-status

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" final-watch

The independent integrity audit is already run automatically. It can be
repeated at any time after completion:

  bash "$HOME/V7_7_GENERALIZATION_SOURCE/v7_7.sh" verify

The finalizer automatically creates real path images and all graphs. The
standalone pictures mode remains available only for an optional second set.

FINAL OUTPUT NAME
-----------------
Folder:
  $HOME/V7_7_GENERALIZATION_GAN_FINAL

Archive:
  $HOME/V7_7_GENERALIZATION_GAN_FINAL.zip

The folder contains the exact selected model, checkpoints, dataset, raw
metrics, raw rollout cases, statistical summaries, charts, matching chart
data, route images, full source code, hashes, and verification records.

SCIENTIFIC NAMING NOTE
----------------------
The protected accuracy-first launcher uses zero adversarial epochs. Its motion
model is therefore a supervised, physics-informed conditional generator. The
separate final discriminator is an audit and does not retroactively make the
accuracy phase adversarial. Describe it as a GAN only if a genuine jointly
adversarial generator-training phase is later enabled and reported.

The launcher refuses an accidental second run, legacy warm start, nonzero hard
mining, ambiguous checkpoint arguments, and output overwrites.
