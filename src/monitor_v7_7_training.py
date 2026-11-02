#!/usr/bin/env python3
"""Print concise, truthful V7.7 training status from its append-only log."""

from pathlib import Path
import argparse
import json
import os


FIELDS = (
    "epoch",
    "phase",
    "generator_learning_rate",
    "train_g_loss",
    "training_probe_deterministic_velocity_accuracy_percent",
    "generator_deterministic_position_accuracy_percent",
    "generator_deterministic_velocity_accuracy_percent",
    "generator_deterministic_full_state_accuracy_percent",
    "generator_deterministic_mean_velocity_error",
    "generator_deterministic_p95_velocity_error",
    "training_validation_velocity_accuracy_gap_percent",
    "minimum_deployable_validation_accuracy_percent",
    "deployable_validation_target_reached",
)


def process_is_running(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def show_row(label, row):
    print(f"\n{label}")
    print("=" * len(label))
    for key in FIELDS:
        print(f"{key}: {row.get(key)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", default=str(Path.home() / "V7_7_training_log.txt")
    )
    parser.add_argument(
        "--pid_file", default=str(Path.home() / "V7_7_training.pid")
    )
    args = parser.parse_args()
    log_path = Path(args.log).expanduser()
    if not log_path.is_file():
        raise SystemExit(f"STOP: training log not found: {log_path}")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('{"epoch"'):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if "selection_score" in row:
            rows.append(row)
    pid_path = Path(args.pid_file).expanduser()
    pid = None
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
    print(f"LOG: {log_path}")
    print(f"PROCESS PID: {pid if pid is not None else 'not recorded'}")
    print(f"PROCESS RUNNING: {process_is_running(pid) if pid is not None else 'unknown'}")
    print(f"VALIDATION EVALUATIONS FOUND: {len(rows)}")
    if not rows:
        print("No completed validation evaluation has been written yet.")
        progress = [line for line in lines if "PROGRESS:" in line]
        if progress:
            print("LATEST PROGRESS:", progress[-1])
        return
    best = max(rows, key=lambda row: float(row["selection_score"]))
    latest = rows[-1]
    show_row("VALIDATION-SELECTED BEST", best)
    show_row("LATEST EVALUATION", latest)
    progress = [line for line in lines if "TRAINING PROGRESS:" in line]
    if progress:
        print("\nLATEST PROGRESS")
        print("===============")
        print(progress[-1])
    if any("STOPPED AFTER TRAINING AS REQUESTED" in line for line in lines):
        print("\nTRAINING PHASE FINISHED: yes")
    elif any("Traceback (most recent call last)" in line for line in lines):
        print("\nTRAINING PHASE FINISHED: stopped by an error; inspect the log tail")
    else:
        print("\nTRAINING PHASE FINISHED: no")
    print(
        "\nCERTIFICATION RULE: position error <= 0.05, velocity error <= 0.05, "
        "and full-state error <= 0.10; all three deterministic validation "
        "accuracies must be at least the requested target."
    )


if __name__ == "__main__":
    main()
