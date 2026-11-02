#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$HOME/lqr_cbf_env/bin/python3"
TRAINER="$SOURCE_DIR/build_v7_7_rrt_sample_gan.py"
RENDERER="$SOURCE_DIR/render_v7_7_path_pictures.py"
GALLERY="$SOURCE_DIR/build_v7_7_results_gallery.py"
FINAL_PIPELINE="$SOURCE_DIR/run_v7_7_final_pipeline.sh"
VERIFIER="$SOURCE_DIR/verify_v7_7_final_package.py"
MONITOR="$SOURCE_DIR/monitor_v7_7_training.py"
DATASET="$HOME/V7_3_RRT_SAMPLE_DATASET_FINAL/rrt_sample_gan_dataset_v7.npz"
OUTPUT_DIR="$HOME/V7_7_GENERALIZATION_GAN_FINAL"
CHECKPOINT="${OUTPUT_DIR}_TRAINING_CHECKPOINT.pt"
PACKAGE_ZIP="${OUTPUT_DIR}.zip"
TRAIN_LOG="$HOME/V7_7_training_log.txt"
FINAL_LOG="$HOME/V7_7_finalization_log.txt"
PREFLIGHT_LOG="$HOME/V7_7_preflight_log.txt"
PID_FILE="$HOME/V7_7_training.pid"

COMMON_ARGS=(
  --dataset "$DATASET"
  --output_dir "$OUTPUT_DIR"
  --supervised_epochs 400
  --adversarial_epochs 0
  --batch_size 512
  --noise_dim 16
  --lr_g 0.0002
  --generator_weight_decay 0.0001
  --d4_augmentation_probability 1.0
  --hard_example_fraction 0
  --residual_only_epochs 0
  --eval_every 5
  --eval_candidates 16
  --eval_candidate_chunk 16
  --generalization_probe_rows 6000
  --minimum_supervised_epochs 80
  --early_stopping_patience_evaluations 8
  --target_accuracy 98
  --final_d_steps 3000
  --final_d_eval_every 150
  --pictures_per_split 0
  --benchmark_cases_per_difficulty 100
  --max_iterations 4500
  --max_nodes 900
  --candidates_per_expand 64
  --children_per_expand 1
  --attempts_per_case 3
  --goal_sample_rate 0.18
  --cpu
)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "STOP: required file was not found: $1" >&2
    exit 1
  fi
}

ensure_idle() {
  if [[ -f "$PID_FILE" ]]; then
    local recorded_pid
    recorded_pid="$(tr -d '[:space:]' < "$PID_FILE")"
    if [[ "$recorded_pid" =~ ^[0-9]+$ ]] && kill -0 "$recorded_pid" 2>/dev/null; then
      echo "STOP: V7.7 is already running with PID $recorded_pid" >&2
      echo "Run: bash \"$SOURCE_DIR/v7_7.sh\" status" >&2
      exit 1
    fi
  fi
}

launch_background() {
  local log_file="$1"
  shift
  ensure_idle
  nohup "$PYTHON" -u "$TRAINER" "${COMMON_ARGS[@]}" "$@" \
    >> "$log_file" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    echo "STARTED SUCCESSFULLY"
    echo "PID: $pid"
    echo "LOG: $log_file"
    echo "STATUS: bash \"$SOURCE_DIR/v7_7.sh\" status"
    echo "LIVE LOG: bash \"$SOURCE_DIR/v7_7.sh\" watch"
  else
    echo "START FAILED. Last log lines:" >&2
    tail -n 40 "$log_file" >&2 || true
    exit 1
  fi
}

launch_final_pipeline() {
  ensure_idle
  require_file "$FINAL_PIPELINE"
  require_file "$GALLERY"
  require_file "$RENDERER"
  require_file "$VERIFIER"
  if [[ -e "$OUTPUT_DIR" ]] || [[ -e "$PACKAGE_ZIP" ]]; then
    echo "STOP: final output already exists; refusing to overwrite it." >&2
    echo "Folder: $OUTPUT_DIR" >&2
    echo "ZIP: $PACKAGE_ZIP" >&2
    exit 1
  fi
  : > "$FINAL_LOG"
  nohup bash "$FINAL_PIPELINE" \
    "$PYTHON" "$TRAINER" "$RENDERER" "$GALLERY" "$VERIFIER" "$OUTPUT_DIR" \
    "${COMMON_ARGS[@]}" \
    >> "$FINAL_LOG" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    echo "FINAL PIPELINE STARTED SUCCESSFULLY"
    echo "PID: $pid"
    echo "LOG: $FINAL_LOG"
    echo "STATUS: bash \"$SOURCE_DIR/v7_7.sh\" final-status"
    echo "LIVE LOG: bash \"$SOURCE_DIR/v7_7.sh\" final-watch"
  else
    echo "FINAL PIPELINE FAILED TO START. Last log lines:" >&2
    tail -n 60 "$FINAL_LOG" >&2 || true
    exit 1
  fi
}

mode="${1:-help}"
case "$mode" in
  preflight)
    require_file "$PYTHON"
    require_file "$TRAINER"
    require_file "$DATASET"
    "$PYTHON" -m py_compile "$TRAINER" "$RENDERER" "$GALLERY" "$VERIFIER" "$MONITOR"
    bash -n "$SOURCE_DIR/v7_7.sh" "$FINAL_PIPELINE"
    "$PYTHON" -u "$GALLERY" --self_test
    "$PYTHON" -u "$TRAINER" "${COMMON_ARGS[@]}" --preflight_only \
      2>&1 | tee "$PREFLIGHT_LOG"
    ;;
  train)
    require_file "$PYTHON"
    require_file "$TRAINER"
    require_file "$DATASET"
    if [[ -e "$CHECKPOINT" ]]; then
      echo "STOP: a V7.7 checkpoint already exists: $CHECKPOINT" >&2
      echo "Use the resume mode; do not restart or delete it." >&2
      exit 1
    fi
    if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "STOP: output directory already contains files: $OUTPUT_DIR" >&2
      exit 1
    fi
    if [[ -e "$PACKAGE_ZIP" ]]; then
      echo "STOP: package ZIP already exists: $PACKAGE_ZIP" >&2
      exit 1
    fi
    : > "$TRAIN_LOG"
    launch_background "$TRAIN_LOG" --stop_after_training
    ;;
  resume)
    require_file "$CHECKPOINT"
    launch_background "$TRAIN_LOG" --resume --stop_after_training
    ;;
  status)
    "$PYTHON" "$MONITOR" --log "$TRAIN_LOG" --pid_file "$PID_FILE"
    ;;
  watch)
    require_file "$TRAIN_LOG"
    tail -n 60 -f "$TRAIN_LOG"
    ;;
  finalize)
    require_file "$CHECKPOINT"
    launch_final_pipeline
    ;;
  final-status)
    if [[ -f "$PID_FILE" ]]; then
      FINAL_PID="$(tr -d '[:space:]' < "$PID_FILE")"
      echo "FINAL PIPELINE PID: $FINAL_PID"
      if [[ "$FINAL_PID" =~ ^[0-9]+$ ]] && kill -0 "$FINAL_PID" 2>/dev/null; then
        echo "FINAL PIPELINE RUNNING: yes"
        ps -p "$FINAL_PID" -o pid,etime,cmd
      else
        echo "FINAL PIPELINE RUNNING: no"
      fi
    else
      echo "FINAL PIPELINE PID: not recorded"
    fi
    if [[ -f "$FINAL_LOG" ]]; then
      echo
      tail -n 30 "$FINAL_LOG"
    else
      echo "FINAL LOG NOT FOUND: $FINAL_LOG"
    fi
    ;;
  final-watch)
    require_file "$FINAL_LOG"
    tail -n 80 -f "$FINAL_LOG"
    ;;
  pictures)
    require_file "$RENDERER"
    if [[ ! -d "$OUTPUT_DIR" ]]; then
      echo "STOP: completed package directory not found: $OUTPUT_DIR" >&2
      exit 1
    fi
    "$PYTHON" -u "$RENDERER" \
      --package_dir "$OUTPUT_DIR" \
      --output_dir "${OUTPUT_DIR}_PATH_PICTURES" \
      --pictures_per_split 15 \
      --cpu
    ;;
  verify)
    require_file "$VERIFIER"
    if [[ ! -d "$OUTPUT_DIR" ]]; then
      echo "STOP: final output folder not found: $OUTPUT_DIR" >&2
      exit 1
    fi
    "$PYTHON" -u "$VERIFIER" --package_dir "$OUTPUT_DIR"
    ;;
  help|--help|-h)
    echo "Usage: bash \"$SOURCE_DIR/v7_7.sh\" MODE"
    echo
    echo "Modes:"
    echo "  preflight  Compile and run all dataset/model safety checks"
    echo "  train      Start a new protected V7.7 training run"
    echo "  resume     Continue an interrupted V7.7 training run"
    echo "  status     Show selected/latest validation statistics"
    echo "  watch      Follow the live training log (Ctrl+C only stops watching)"
    echo "  finalize   After 98% validation, run held-out evaluation and package"
    echo "  final-status  Show final-evaluation stage, elapsed time, and ETA"
    echo "  final-watch   Follow the complete final-pipeline log"
    echo "  pictures   Separately render real package rollout paths"
    echo "  verify     Independently recheck metrics, rows, hashes, and ZIP"
    ;;
  *)
    echo "STOP: unknown mode: $mode" >&2
    echo "Run: bash \"$SOURCE_DIR/v7_7.sh\" help" >&2
    exit 2
    ;;
esac
