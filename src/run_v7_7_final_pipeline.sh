#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "STOP: internal final-pipeline invocation is incomplete." >&2
  exit 2
fi

PYTHON="$1"
TRAINER="$2"
RENDERER="$3"
GALLERY="$4"
VERIFIER="$5"
OUTPUT_DIR="$6"
shift 6
TRAIN_ARGS=("$@")

PIPELINE_STARTED="$(date +%s)"
STAGE_STARTED="$PIPELINE_STARTED"

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  if (( hours > 0 )); then
    printf '%dh %02dm %02ds' "$hours" "$minutes" "$seconds"
  elif (( minutes > 0 )); then
    printf '%dm %02ds' "$minutes" "$seconds"
  else
    printf '%ds' "$seconds"
  fi
}

stage_start() {
  STAGE_STARTED="$(date +%s)"
  echo
  echo "============================================================"
  echo "FINAL PIPELINE STAGE $1/4: $2"
  echo "Stage ETA: calculated from real completed cases by the program"
  echo "============================================================"
}

stage_done() {
  local now
  now="$(date +%s)"
  echo "STAGE FINISHED: $1"
  echo "Stage elapsed: $(format_seconds $((now - STAGE_STARTED)))"
  echo "Total elapsed: $(format_seconds $((now - PIPELINE_STARTED)))"
}

for required in "$PYTHON" "$TRAINER" "$RENDERER" "$GALLERY" "$VERIFIER"; do
  if [[ ! -f "$required" ]]; then
    echo "STOP: required final-pipeline file was not found: $required" >&2
    exit 1
  fi
done

if [[ -e "$OUTPUT_DIR" ]] || [[ -e "${OUTPUT_DIR}.zip" ]]; then
  echo "STOP: final output already exists; nothing will be overwritten." >&2
  echo "Folder: $OUTPUT_DIR" >&2
  echo "ZIP: ${OUTPUT_DIR}.zip" >&2
  exit 1
fi

stage_start 1 "held-out evaluation, classifier audit, and rollout benchmarks"
"$PYTHON" -u "$TRAINER" "${TRAIN_ARGS[@]}" --resume
stage_done "verified final evaluation and base package"

if [[ ! -f "$OUTPUT_DIR/BUILD_COMPLETED_SUCCESSFULLY.txt" ]]; then
  echo "STOP: trainer returned without a verified final package." >&2
  exit 1
fi

PATH_STAGE="$OUTPUT_DIR/.path_image_staging"
stage_start 2 "real easy/medium/hard route-image rendering"
"$PYTHON" -u "$RENDERER" \
  --package_dir "$OUTPUT_DIR" \
  --output_dir "$PATH_STAGE" \
  --pictures_per_split 15 \
  --cpu
stage_done "45 real rollout path images"

stage_start 3 "numbered charts, matching CSV data, integrity manifest, and ZIP"
"$PYTHON" -u "$GALLERY" \
  --package_dir "$OUTPUT_DIR" \
  --path_images_dir "$PATH_STAGE" \
  --consume_path_images \
  --overwrite
stage_done "results gallery and final archive"

stage_start 4 "independent disk, benchmark, manifest, and ZIP hash audit"
"$PYTHON" -u "$VERIFIER" --package_dir "$OUTPUT_DIR"
stage_done "independent final integrity audit"

if [[ ! -f "${OUTPUT_DIR}.zip" ]]; then
  echo "STOP: final ZIP was not created." >&2
  exit 1
fi

echo
echo "EVERYTHING FINISHED SUCCESSFULLY"
echo "Final folder: $OUTPUT_DIR"
echo "Final ZIP: ${OUTPUT_DIR}.zip"
echo "Total elapsed: $(format_seconds $(($(date +%s) - PIPELINE_STARTED)))"
