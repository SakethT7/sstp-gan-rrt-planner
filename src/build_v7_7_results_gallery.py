#!/usr/bin/env python3
"""Build a publication-quality V7.7 results gallery from final real outputs.

This script never invents values. Every chart is created from FINAL_METRICS.json,
the saved validation history, the final-discriminator history, or the exact
rollout CSV files produced by the protected finalizer. Each figure receives a
matching CSV table and an indexed caption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "v7_7_matplotlib_cache")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GALLERY_VERSION = "1.0.0-real-data-only"
BLUE = "#00529B"
LIGHT_BLUE = "#6CACE4"
DARK_BLUE = "#0021A5"
GRAY = "#6D6E71"
LIGHT_GRAY = "#C9CDD2"
DARK_GRAY = "#252525"
PALETTE = (DARK_BLUE, BLUE, LIGHT_BLUE, GRAY, LIGHT_GRAY, DARK_GRAY)


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{seconds:.1f}s"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path, value):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def as_float(value, default=float("nan")):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def clean_axes(ax, grid_axis="y"):
    ax.set_facecolor("white")
    ax.grid(True, axis=grid_axis, color="#D8DCE1", linewidth=0.75, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#60656B")
    ax.spines["bottom"].set_color("#60656B")
    ax.tick_params(colors=DARK_GRAY, labelsize=9)


def annotate_bars(ax, bars, decimals=2, suffix=""):
    for bar in bars:
        height = float(bar.get_height())
        if not math.isfinite(height):
            continue
        ax.annotate(
            f"{height:.{decimals}f}{suffix}",
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=DARK_GRAY,
        )


class Gallery:
    def __init__(self, root):
        self.root = Path(root)
        self.figures = self.root / "01_FIGURES"
        self.data = self.root / "02_FIGURE_DATA"
        self.tables = self.root / "03_SUMMARY_TABLES"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        self.index = []

    @property
    def next_number(self):
        return len(self.index) + 1

    def save(self, fig, slug, title, caption, rows):
        number = self.next_number
        stem = f"Figure_{number:02d}_{slug}"
        png = self.figures / f"{stem}.png"
        svg = self.figures / f"{stem}.svg"
        csv_path = self.data / f"{stem}.csv"
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        write_csv(csv_path, rows)
        self.index.append({
            "figure": f"Figure {number}",
            "title": title,
            "caption": caption,
            "png": str(png.relative_to(self.root)),
            "svg": str(svg.relative_to(self.root)),
            "data_csv": str(csv_path.relative_to(self.root)),
        })

    def grouped_bar(
        self, slug, title, caption, categories, series, rows,
        ylabel, ylim=None, threshold=None, decimals=2, percent=False,
    ):
        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        x = np.arange(len(categories), dtype=float)
        count = max(1, len(series))
        width = min(0.24, 0.78 / count)
        offsets = (np.arange(count) - (count - 1) / 2.0) * width
        for index, (label, values) in enumerate(series.items()):
            bars = ax.bar(
                x + offsets[index], values, width=width * 0.92,
                label=label, color=PALETTE[index % len(PALETTE)],
                edgecolor="white", linewidth=0.5,
            )
            annotate_bars(
                ax, bars, decimals=decimals, suffix="%" if percent else ""
            )
        if threshold is not None:
            ax.axhline(
                float(threshold), color=DARK_GRAY, linestyle="--", linewidth=1.3,
                label=f"Target ({float(threshold):g}%)",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel, color=DARK_GRAY)
        ax.set_title(title, fontsize=15, weight="semibold", color=DARK_GRAY, pad=14)
        if ylim is not None:
            ax.set_ylim(*ylim)
        clean_axes(ax)
        ax.legend(frameon=False, fontsize=9, ncol=min(4, len(series) + 1))
        self.save(fig, slug, title, caption, rows)

    def line(
        self, slug, title, caption, x, series, rows, xlabel, ylabel,
        ylim=None, threshold=None, log_y=False,
    ):
        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        for index, (label, values) in enumerate(series.items()):
            ax.plot(
                x, values, marker="o", markersize=3.8, linewidth=2.0,
                color=PALETTE[index % len(PALETTE)], label=label,
            )
        if threshold is not None:
            ax.axhline(
                float(threshold), color=DARK_GRAY, linestyle="--", linewidth=1.3,
                label=f"Target ({float(threshold):g}%)",
            )
        if log_y:
            positive = [
                value for values in series.values() for value in values
                if math.isfinite(value) and value > 0
            ]
            if positive:
                ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, color=DARK_GRAY)
        ax.set_ylabel(ylabel, color=DARK_GRAY)
        ax.set_title(title, fontsize=15, weight="semibold", color=DARK_GRAY, pad=14)
        clean_axes(ax, grid_axis="both")
        ax.legend(frameon=False, fontsize=9)
        self.save(fig, slug, title, caption, rows)

    def single_bar(
        self, slug, title, caption, categories, values, rows, ylabel,
        ylim=None, decimals=2, percent=False, colors=None,
    ):
        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        colors = colors or [PALETTE[i % len(PALETTE)] for i in range(len(values))]
        bars = ax.bar(
            np.arange(len(categories)), values, color=colors,
            edgecolor="white", linewidth=0.5, width=0.66,
        )
        annotate_bars(ax, bars, decimals=decimals, suffix="%" if percent else "")
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylabel(ylabel, color=DARK_GRAY)
        ax.set_title(title, fontsize=15, weight="semibold", color=DARK_GRAY, pad=14)
        if ylim is not None:
            ax.set_ylim(*ylim)
        clean_axes(ax)
        self.save(fig, slug, title, caption, rows)


def metric_rows(metrics, keys):
    rows = []
    for split in ("training", "validation", "testing"):
        row = {"split": split}
        for output_name, metric_key in keys:
            row[output_name] = float(metrics[split][metric_key])
        rows.append(row)
    return rows


def add_metric_figures(gallery, metrics, target_accuracy):
    splits = ["Training", "Validation", "Testing"]

    rows = metric_rows(metrics, [
        ("position_accuracy_percent", "generator_deterministic_position_accuracy_percent"),
        ("velocity_accuracy_percent", "generator_deterministic_velocity_accuracy_percent"),
        ("full_state_accuracy_percent", "generator_deterministic_full_state_accuracy_percent"),
    ])
    gallery.grouped_bar(
        "deterministic_accuracy_by_split",
        "Deployable deterministic accuracy by data split",
        "Deterministic zero-noise accuracy on the training, validation, and untouched testing splits. The dashed line is the preregistered deployment target.",
        splits,
        {
            "Position": [row["position_accuracy_percent"] for row in rows],
            "Velocity": [row["velocity_accuracy_percent"] for row in rows],
            "Full state": [row["full_state_accuracy_percent"] for row in rows],
        },
        rows,
        "Accuracy (%)", ylim=(95.0, 100.6), threshold=target_accuracy,
        decimals=2, percent=True,
    )

    rows = metric_rows(metrics, [
        ("velocity_accuracy_at_0_025_percent", "generator_deterministic_velocity_accuracy_at_0_025_percent"),
        ("velocity_accuracy_at_0_05_percent", "generator_deterministic_velocity_accuracy_at_0_05_percent"),
        ("velocity_accuracy_at_0_10_percent", "generator_deterministic_velocity_accuracy_at_0_10_percent"),
    ])
    minimum = min(value for row in rows for key, value in row.items() if key != "split")
    gallery.grouped_bar(
        "velocity_accuracy_thresholds",
        "Velocity accuracy at three error tolerances",
        "Percentage of deterministic predictions whose two-dimensional normalized velocity error is within each stated tolerance.",
        splits,
        {
            "Error ≤ 0.025": [row["velocity_accuracy_at_0_025_percent"] for row in rows],
            "Error ≤ 0.050": [row["velocity_accuracy_at_0_05_percent"] for row in rows],
            "Error ≤ 0.100": [row["velocity_accuracy_at_0_10_percent"] for row in rows],
        },
        rows,
        "Accuracy (%)", ylim=(max(0.0, math.floor(minimum) - 1.0), 100.6),
        decimals=2, percent=True,
    )

    rows = metric_rows(metrics, [
        ("mean_velocity_error", "generator_deterministic_mean_velocity_error"),
        ("p95_velocity_error", "generator_deterministic_p95_velocity_error"),
    ])
    gallery.grouped_bar(
        "velocity_error_by_split",
        "Deterministic velocity error by data split",
        "Mean and 95th-percentile two-dimensional normalized velocity error. Lower values are better.",
        splits,
        {
            "Mean error": [row["mean_velocity_error"] for row in rows],
            "95th percentile": [row["p95_velocity_error"] for row in rows],
        },
        rows, "Normalized velocity error", decimals=4,
    )

    rows = metric_rows(metrics, [
        ("mean_position_error", "generator_deterministic_mean_position_error"),
        ("p95_position_error", "generator_deterministic_p95_position_error"),
    ])
    gallery.grouped_bar(
        "position_error_by_split",
        "Deterministic position error by data split",
        "Mean and 95th-percentile two-dimensional normalized position error. Lower values are better.",
        splits,
        {
            "Mean error": [row["mean_position_error"] for row in rows],
            "95th percentile": [row["p95_position_error"] for row in rows],
        },
        rows, "Normalized position error", decimals=6,
    )

    rows = metric_rows(metrics, [
        ("full_state_mse", "generator_deterministic_full_state_mse_loss"),
    ])
    gallery.single_bar(
        "full_state_mse_by_split",
        "Deterministic full-state mean squared error",
        "Mean squared component error over position and velocity for each split. Lower values are better.",
        splits, [row["full_state_mse"] for row in rows], rows,
        "Mean squared error", decimals=8,
    )

    rows = metric_rows(metrics, [
        ("control_increment_accuracy_percent", "generator_deterministic_control_increment_accuracy_at_0_05_percent"),
        ("mean_control_increment_error", "generator_deterministic_mean_control_increment_error"),
        ("p95_control_increment_error", "generator_deterministic_p95_control_increment_error"),
    ])
    gallery.single_bar(
        "control_increment_accuracy",
        "Control-increment accuracy by data split",
        "Percentage of deterministic one-step control increments with normalized vector error no greater than 0.05.",
        splits, [row["control_increment_accuracy_percent"] for row in rows], rows,
        "Accuracy (%)", ylim=(95.0, 100.6), decimals=2, percent=True,
    )
    gallery.grouped_bar(
        "control_increment_error",
        "Control-increment error by data split",
        "Mean and 95th-percentile normalized control-increment error. Lower values are better.",
        splits,
        {
            "Mean error": [row["mean_control_increment_error"] for row in rows],
            "95th percentile": [row["p95_control_increment_error"] for row in rows],
        },
        rows, "Normalized control-increment error", decimals=4,
    )

    rows = metric_rows(metrics, [
        ("deterministic_velocity_accuracy_percent", "generator_deterministic_velocity_accuracy_percent"),
        ("bestof_velocity_accuracy_percent", "generator_bestof_velocity_accuracy_percent"),
    ])
    gallery.grouped_bar(
        "deterministic_vs_oracle_velocity",
        "Deployable output versus best-of oracle diagnostic",
        "The deterministic value is deployable. Best-of uses the true label after generation and is shown only as an oracle diagnostic.",
        splits,
        {
            "Deterministic": [row["deterministic_velocity_accuracy_percent"] for row in rows],
            "Best-of oracle": [row["bestof_velocity_accuracy_percent"] for row in rows],
        },
        rows, "Velocity accuracy (%)", ylim=(95.0, 100.6),
        decimals=2, percent=True,
    )

    rows = metric_rows(metrics, [
        ("mean_candidate_deviation", "mean_random_candidate_deviation_from_deterministic_state"),
        ("maximum_candidate_deviation", "maximum_random_candidate_deviation_from_deterministic_state"),
    ])
    gallery.grouped_bar(
        "candidate_diversity",
        "Random-candidate deviation from deterministic steering",
        "Mean and maximum state deviation produced by noisy candidates relative to the zero-noise deployable output.",
        splits,
        {
            "Mean deviation": [row["mean_candidate_deviation"] for row in rows],
            "Maximum deviation": [row["maximum_candidate_deviation"] for row in rows],
        },
        rows, "Normalized full-state deviation", decimals=5,
    )


def numeric_history_rows(history):
    result = []
    for row in history:
        if "epoch" not in row or "selection_score" not in row:
            continue
        copied = dict(row)
        copied["epoch"] = int(float(copied["epoch"]))
        result.append(copied)
    return sorted(result, key=lambda row: row["epoch"])


def add_training_figures(gallery, history, target_accuracy):
    history = numeric_history_rows(history)
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    rows = [{
        "epoch": row["epoch"],
        "position_accuracy_percent": as_float(row.get("generator_deterministic_position_accuracy_percent")),
        "velocity_accuracy_percent": as_float(row.get("generator_deterministic_velocity_accuracy_percent")),
        "full_state_accuracy_percent": as_float(row.get("generator_deterministic_full_state_accuracy_percent")),
    } for row in history]
    gallery.line(
        "validation_accuracy_progression",
        "Validation accuracy during training",
        "Validation checkpoint metrics used for honest model selection; testing data are absent from this progression.",
        epochs,
        {
            "Position": [row["position_accuracy_percent"] for row in rows],
            "Velocity": [row["velocity_accuracy_percent"] for row in rows],
            "Full state": [row["full_state_accuracy_percent"] for row in rows],
        },
        rows, "Epoch", "Accuracy (%)", ylim=(50.0, 101.0),
        threshold=target_accuracy,
    )

    rows = [{
        "epoch": row["epoch"],
        "training_probe_velocity_accuracy_percent": as_float(row.get("training_probe_deterministic_velocity_accuracy_percent")),
        "validation_velocity_accuracy_percent": as_float(row.get("generator_deterministic_velocity_accuracy_percent")),
    } for row in history]
    gallery.line(
        "training_validation_velocity_accuracy",
        "Training-probe and validation velocity accuracy",
        "Agreement between the fixed training probe and held-out validation reveals whether accuracy gains generalize instead of memorizing environments.",
        epochs,
        {
            "Training probe": [row["training_probe_velocity_accuracy_percent"] for row in rows],
            "Validation": [row["validation_velocity_accuracy_percent"] for row in rows],
        },
        rows, "Epoch", "Velocity accuracy (%)", ylim=(50.0, 101.0),
        threshold=target_accuracy,
    )

    rows = [{
        "epoch": row["epoch"],
        "mean_velocity_error": as_float(row.get("generator_deterministic_mean_velocity_error")),
        "p95_velocity_error": as_float(row.get("generator_deterministic_p95_velocity_error")),
    } for row in history]
    gallery.line(
        "validation_velocity_error_progression",
        "Validation velocity error during training",
        "Mean and 95th-percentile deterministic velocity errors. Lower values indicate more accurate tail behavior.",
        epochs,
        {
            "Mean error": [row["mean_velocity_error"] for row in rows],
            "95th percentile": [row["p95_velocity_error"] for row in rows],
        },
        rows, "Epoch", "Normalized velocity error",
    )

    rows = [{
        "epoch": row["epoch"],
        "generator_training_loss": as_float(row.get("train_g_loss")),
        "validation_full_state_mse": as_float(row.get("generator_deterministic_full_state_mse_loss")),
    } for row in history]
    gallery.line(
        "loss_progression",
        "Generator training loss and validation state error",
        "The generator loss is a weighted training objective; validation full-state MSE is shown separately on the same logarithmic scale.",
        epochs,
        {
            "Weighted generator loss": [row["generator_training_loss"] for row in rows],
            "Validation full-state MSE": [row["validation_full_state_mse"] for row in rows],
        },
        rows, "Epoch", "Loss (log scale)", log_y=True,
    )

    rows = [{
        "epoch": row["epoch"],
        "generator_learning_rate": as_float(row.get("generator_learning_rate")),
    } for row in history]
    gallery.line(
        "learning_rate_schedule",
        "Generator learning-rate schedule",
        "Learning rate recorded at every validation checkpoint.",
        epochs,
        {"Generator learning rate": [row["generator_learning_rate"] for row in rows]},
        rows, "Epoch", "Learning rate",
    )

    rows = [{
        "epoch": row["epoch"],
        "training_minus_validation_velocity_accuracy_points": as_float(
            row.get("training_validation_velocity_accuracy_gap_percent")
        ),
    } for row in history]
    gallery.line(
        "generalization_gap",
        "Velocity-accuracy generalization gap",
        "Training-probe accuracy minus validation accuracy in percentage points. Values near zero indicate matched performance.",
        epochs,
        {"Training − validation": [row["training_minus_validation_velocity_accuracy_points"] for row in rows]},
        rows, "Epoch", "Accuracy gap (percentage points)", threshold=0.0,
    )


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def mean_field(rows, field, solved_only=False):
    values = []
    for row in rows:
        if solved_only and not as_bool(row.get("solved")):
            continue
        value = as_float(row.get(field))
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def rollout_difficulty_table(rows, field, solved_only=False, percent_mode=None):
    table = []
    groups = group_rows(rows, ("split", "difficulty_bucket"))
    for split in ("training", "validation", "testing"):
        for difficulty in ("easy", "medium", "hard"):
            current = groups.get((split, difficulty), [])
            if percent_mode == "solve":
                value = 100.0 * sum(as_bool(row.get("solved")) for row in current) / max(1, len(current))
            elif percent_mode == "gan_share":
                accepted = sum(int(as_float(row.get("accepted_total"), 0.0)) for row in current)
                gan = sum(int(as_float(row.get("gan_accepted_total"), 0.0)) for row in current)
                value = 100.0 * gan / max(1, accepted)
            else:
                value = mean_field(current, field, solved_only=solved_only)
            table.append({
                "split": split,
                "difficulty": difficulty,
                "cases": len(current),
                "value": value,
            })
    return table


def difficulty_series(table):
    return {
        split.title(): [
            next(
                row["value"] for row in table
                if row["split"] == split and row["difficulty"] == difficulty
            )
            for difficulty in ("easy", "medium", "hard")
        ]
        for split in ("training", "validation", "testing")
    }


def add_rollout_figures(gallery, rollout_rows, classical_rows, comparison):
    difficulties = ["Easy", "Medium", "Hard"]
    specifications = [
        (
            "rollout_solve_rate", "Collision-free planning solve rate",
            "Solve rate across valid, nontrivial cases. Invalid starts/goals and cases already within the goal radius are excluded by policy.",
            "", "Solve rate (%)", False, "solve", (0.0, 105.0), 2, True,
        ),
        (
            "rollout_runtime", "Planner runtime by difficulty",
            "Mean measured wall-clock runtime per benchmark case, including all attempts.",
            "runtime_seconds", "Mean runtime (s)", False, None, None, 3, False,
        ),
        (
            "rollout_tree_nodes", "Tree size by difficulty",
            "Mean number of nodes in the selected planning attempt.",
            "nodes", "Mean tree nodes", False, None, None, 1, False,
        ),
        (
            "rollout_iterations", "Planner iterations by difficulty",
            "Mean number of iterations in the selected planning attempt.",
            "iterations", "Mean iterations", False, None, None, 1, False,
        ),
        (
            "rollout_path_cost", "Solved-path length by difficulty",
            "Mean Euclidean path length among solved cases only; this is not the LQR objective value.",
            "path_cost", "Mean solved path length", True, None, None, 3, False,
        ),
        (
            "rollout_goal_distance", "Final goal distance by difficulty",
            "Mean distance from the selected terminal node to the goal. Lower values are better.",
            "best_distance_to_goal", "Mean final goal distance", False, None, None, 4, False,
        ),
        (
            "gan_node_share", "Generator share of accepted tree nodes",
            "Percentage of accepted nodes produced by generator proposals rather than the conditional safety fallback.",
            "", "Generator share (%)", False, "gan_share", (0.0, 105.0), 2, True,
        ),
    ]
    for slug, title, caption, field, ylabel, solved_only, mode, ylim, decimals, percent in specifications:
        table = rollout_difficulty_table(
            rollout_rows, field, solved_only=solved_only, percent_mode=mode
        )
        gallery.grouped_bar(
            slug, title, caption, difficulties, difficulty_series(table), table,
            ylabel, ylim=ylim, decimals=decimals, percent=percent,
        )

    split_groups = group_rows(rollout_rows, ("split",))
    split_names = ["Training", "Validation", "Testing"]
    rejection_fields = (
        ("Collision/boundary", "collision_boundary_rejections"),
        ("Duplicate", "duplicate_rejections"),
        ("Nonfinite", "nonfinite_candidate_rejections"),
        ("Free-sample failure", "free_sample_failures"),
    )
    rows = []
    series = {}
    for label, field in rejection_fields:
        values = []
        for split in ("training", "validation", "testing"):
            value = mean_field(split_groups.get((split,), []), field)
            values.append(value)
            rows.append({"split": split, "rejection_type": label, "mean_count": value})
        series[label] = values
    gallery.grouped_bar(
        "planner_rejections",
        "Mean rejected proposals per benchmark case",
        "Safety and validity rejections are counted rather than hidden. Collision/boundary rejections include proposals rejected before tree insertion.",
        split_names, series, rows, "Mean rejection count", decimals=1,
    )

    fields = (
        ("Attempts", "attempts_executed"),
        ("Total iterations", "total_iterations_across_attempts"),
    )
    rows = []
    series = {}
    for label, field in fields:
        values = []
        for split in ("training", "validation", "testing"):
            value = mean_field(split_groups.get((split,), []), field)
            values.append(value)
            rows.append({"split": split, "quantity": label, "mean": value})
        series[label] = values
    gallery.grouped_bar(
        "planner_effort",
        "Planning effort across all attempts",
        "Mean attempts and cumulative iterations expended per benchmark case.",
        split_names, series, rows, "Mean count", decimals=1,
    )

    fields = (
        ("Generated candidates", "total_generated_candidates_across_attempts"),
        ("Accepted nodes", "total_accepted_nodes_across_attempts"),
    )
    rows = []
    series = {}
    for label, field in fields:
        values = []
        for split in ("training", "validation", "testing"):
            value = mean_field(split_groups.get((split,), []), field)
            values.append(value)
            rows.append({"split": split, "quantity": label, "mean": value})
        series[label] = values
    gallery.grouped_bar(
        "generated_and_accepted_nodes",
        "Generated candidates and accepted nodes",
        "Mean proposal volume and accepted tree growth across all attempts.",
        split_names, series, rows, "Mean count", decimals=1,
    )

    if classical_rows and comparison:
        gan_solve = as_float(comparison.get("gan_solve_rate_percent"))
        classical_solve = as_float(comparison.get("classical_solve_rate_percent"))
        rows = [
            {"planner": "Generator-steered RRT", "solve_rate_percent": gan_solve},
            {"planner": "Classical direct-steering RRT", "solve_rate_percent": classical_solve},
        ]
        gallery.single_bar(
            "paired_solve_rate",
            "Paired testing solve-rate comparison",
            "Both planners are evaluated on identical held-out testing rows with matched sampling, limits, attempts, and collision checks.",
            ["Generator-steered", "Classical"], [gan_solve, classical_solve], rows,
            "Solve rate (%)", ylim=(0.0, 105.0), decimals=2, percent=True,
            colors=[DARK_BLUE, GRAY],
        )

        gan_runtime = as_float(comparison.get("mean_gan_runtime_seconds_all_cases"))
        classical_runtime = as_float(comparison.get("mean_classical_runtime_seconds_all_cases"))
        rows = [
            {"planner": "Generator-steered RRT", "mean_runtime_seconds": gan_runtime},
            {"planner": "Classical direct-steering RRT", "mean_runtime_seconds": classical_runtime},
        ]
        gallery.single_bar(
            "paired_runtime",
            "Paired testing runtime comparison",
            "Mean measured wall-clock runtime across the same held-out testing cases.",
            ["Generator-steered", "Classical"], [gan_runtime, classical_runtime], rows,
            "Mean runtime (s)", decimals=3, colors=[DARK_BLUE, GRAY],
        )

        gan_path = as_float(comparison.get("mean_gan_path_cost_on_both_solved"))
        classical_path = as_float(comparison.get("mean_classical_path_cost_on_both_solved"))
        if math.isfinite(gan_path) and math.isfinite(classical_path):
            rows = [
                {"planner": "Generator-steered RRT", "mean_path_length": gan_path},
                {"planner": "Classical direct-steering RRT", "mean_path_length": classical_path},
            ]
            gallery.single_bar(
                "paired_path_length",
                "Paired path length when both planners solve",
                "Mean Euclidean path length restricted to testing cases solved by both planners.",
                ["Generator-steered", "Classical"], [gan_path, classical_path], rows,
                "Mean path length", decimals=3, colors=[DARK_BLUE, GRAY],
            )

        outcome_keys = (
            ("Both solved", "both_solved"),
            ("Generator only", "gan_only_solved"),
            ("Classical only", "classical_only_solved"),
            ("Neither solved", "neither_solved"),
        )
        outcome_values = [int(comparison.get(key, 0)) for _, key in outcome_keys]
        rows = [
            {"paired_outcome": label, "cases": value}
            for (label, _), value in zip(outcome_keys, outcome_values)
        ]
        gallery.single_bar(
            "paired_outcomes",
            "Paired testing outcomes",
            "Counts of matched held-out cases categorized by which planner reached the goal.",
            [label for label, _ in outcome_keys], outcome_values, rows,
            "Testing cases", decimals=0,
            colors=[DARK_BLUE, BLUE, GRAY, LIGHT_GRAY],
        )


def add_discriminator_figure(gallery, rows):
    if not rows:
        return
    clean = []
    for row in rows:
        step = as_float(row.get("step"))
        if not math.isfinite(step):
            continue
        clean.append({
            "step": int(step),
            "training_accuracy_percent": as_float(row.get("training_accuracy")),
            "validation_accuracy_percent": as_float(row.get("validation_accuracy")),
            "training_bce_loss": as_float(row.get("training_loss")),
            "validation_bce_loss": as_float(row.get("validation_loss")),
        })
    if not clean:
        return
    clean.sort(key=lambda row: row["step"])
    x = [row["step"] for row in clean]
    gallery.line(
        "final_discriminator_accuracy",
        "Final discriminator accuracy progression",
        "Training and validation classification accuracy for the frozen-generator audit. Approximately 50% is a balance diagnostic, not motion accuracy.",
        x,
        {
            "Training": [row["training_accuracy_percent"] for row in clean],
            "Validation": [row["validation_accuracy_percent"] for row in clean],
        },
        clean, "Classifier step", "Accuracy (%)", ylim=(0.0, 100.0), threshold=50.0,
    )
    gallery.line(
        "final_discriminator_loss",
        "Final discriminator BCE loss progression",
        "Binary cross-entropy for real-versus-generated classification with the generator frozen.",
        x,
        {
            "Training": [row["training_bce_loss"] for row in clean],
            "Validation": [row["validation_bce_loss"] for row in clean],
        },
        clean, "Classifier step", "Binary cross-entropy loss",
    )


def flatten_final_metrics(metrics):
    rows = []
    for split in ("training", "validation", "testing"):
        for metric, value in sorted(metrics[split].items()):
            if isinstance(value, (int, float, bool, str)) or value is None:
                rows.append({"split": split, "metric": metric, "value": value})
    return rows


def write_text_summary(path, metrics, verification, figure_count, path_image_count):
    lines = [
        "V7.7 FINAL RESULTS GALLERY",
        "==========================",
        "",
        "All figures are generated from recorded final metrics and rollout data.",
        "No values, paths, successes, or failures are fabricated.",
        "",
    ]
    for split in ("training", "validation", "testing"):
        row = metrics[split]
        lines.extend([
            split.upper(),
            f"  deterministic position accuracy: {row['generator_deterministic_position_accuracy_percent']:.6f}%",
            f"  deterministic velocity accuracy: {row['generator_deterministic_velocity_accuracy_percent']:.6f}%",
            f"  deterministic full-state accuracy: {row['generator_deterministic_full_state_accuracy_percent']:.6f}%",
            f"  mean velocity error: {row['generator_deterministic_mean_velocity_error']:.9f}",
            f"  p95 velocity error: {row['generator_deterministic_p95_velocity_error']:.9f}",
            "",
        ])
    lines.extend([
        f"Package verification status: {verification.get('status', 'unknown')}",
        f"Testing rollout solve rate: {verification.get('testing_rollout_solve_rate_percent')}%",
        f"Generator share of accepted nodes: {verification.get('gan_share_of_accepted_nodes_across_benchmark_percent')}%",
        f"Numbered data figures: {figure_count}",
        f"Real rollout path images: {path_image_count}",
        "",
        "Interpretation boundary:",
        "Accuracy is the percentage of rows within a stated error threshold; it is not exact equality.",
        "Best-of is an oracle diagnostic. Deterministic metrics are the deployable metrics.",
        "The final discriminator is a distribution-balance audit, not a motion-accuracy score.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_manifest_and_zip(package):
    package = Path(package)
    manifest_path = package / "FILE_MANIFEST_SHA256.json"
    manifest = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        manifest[str(path.relative_to(package))] = {
            "sha256": sha256_file(path),
            "size": int(path.stat().st_size),
        }
    atomic_json(manifest_path, manifest)
    temporary_base = package.parent / f"{package.name}_REPACK_TEMP"
    temporary_zip = Path(str(temporary_base) + ".zip")
    if temporary_zip.exists():
        temporary_zip.unlink()
    shutil.make_archive(
        str(temporary_base), "zip", root_dir=package.parent, base_dir=package.name
    )
    final_zip = Path(str(package) + ".zip")
    temporary_zip.replace(final_zip)
    return final_zip, manifest


def count_path_images(path_root):
    path_root = Path(path_root)
    return len(list(path_root.rglob("*.png"))) if path_root.is_dir() else 0


def validate_required(package):
    required = [
        package / "FINAL_METRICS.json",
        package / "verification_report.json",
        package / "05_MODEL_AND_DATASET_FILES" / "model" / "training_history.csv",
        package / "04_OVERALL_BENCHMARK" / "02_OVERALL_BENCHMARK_DATA" / "all_rollout_cases.csv",
        package / "05_MODEL_AND_DATASET_FILES" / "model" / "v7_7_invariant_constraint_rrt_sample_gan_model.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Final package is incomplete; missing: " + "; ".join(missing))
    return required


def run_self_test():
    with tempfile.TemporaryDirectory(prefix="v7_7_gallery_test_") as directory:
        gallery = Gallery(Path(directory) / "gallery")
        mock_metrics = {}
        for split_index, split in enumerate(("training", "validation", "testing")):
            offset = 0.02 * split_index
            mock_metrics[split] = {
                "generator_deterministic_position_accuracy_percent": 100.0,
                "generator_deterministic_velocity_accuracy_percent": 99.9 - offset,
                "generator_deterministic_full_state_accuracy_percent": 100.0,
                "generator_deterministic_velocity_accuracy_at_0_025_percent": 99.5 - offset,
                "generator_deterministic_velocity_accuracy_at_0_05_percent": 99.9 - offset,
                "generator_deterministic_velocity_accuracy_at_0_10_percent": 100.0,
                "generator_deterministic_mean_velocity_error": 0.002 + offset / 100.0,
                "generator_deterministic_p95_velocity_error": 0.008 + offset / 100.0,
                "generator_deterministic_mean_position_error": 0.00002,
                "generator_deterministic_p95_position_error": 0.00008,
                "generator_deterministic_full_state_mse_loss": 0.000004,
                "generator_deterministic_control_increment_accuracy_at_0_05_percent": 99.8,
                "generator_deterministic_mean_control_increment_error": 0.0022,
                "generator_deterministic_p95_control_increment_error": 0.0085,
                "generator_bestof_velocity_accuracy_percent": 99.95,
                "mean_random_candidate_deviation_from_deterministic_state": 0.0003,
                "maximum_random_candidate_deviation_from_deterministic_state": 0.007,
            }
        mock_history = []
        for epoch, accuracy, loss in ((0, 60.0, 1.0), (5, 99.0, 0.05), (10, 99.9, 0.01)):
            mock_history.append({
                "epoch": epoch,
                "selection_score": epoch + 1,
                "generator_deterministic_position_accuracy_percent": 100.0,
                "generator_deterministic_velocity_accuracy_percent": accuracy,
                "generator_deterministic_full_state_accuracy_percent": min(100.0, accuracy + 0.1),
                "training_probe_deterministic_velocity_accuracy_percent": min(100.0, accuracy + 0.05),
                "generator_deterministic_mean_velocity_error": loss / 10.0,
                "generator_deterministic_p95_velocity_error": loss / 4.0,
                "train_g_loss": loss,
                "generator_deterministic_full_state_mse_loss": loss / 20.0,
                "generator_learning_rate": 0.0002,
                "training_validation_velocity_accuracy_gap_percent": 0.05,
            })
        mock_rollouts = []
        for split_index, split in enumerate(("training", "validation", "testing")):
            for difficulty_index, difficulty in enumerate(("easy", "medium", "hard")):
                for case in range(2):
                    mock_rollouts.append({
                        "split": split,
                        "difficulty_bucket": difficulty,
                        "solved": True,
                        "runtime_seconds": 0.2 + 0.1 * difficulty_index,
                        "nodes": 20 + 10 * difficulty_index,
                        "iterations": 25 + 12 * difficulty_index,
                        "path_cost": 0.8 + 0.2 * difficulty_index,
                        "best_distance_to_goal": 0.01,
                        "accepted_total": 20,
                        "gan_accepted_total": 18,
                        "collision_boundary_rejections": 5 + difficulty_index,
                        "duplicate_rejections": 2,
                        "nonfinite_candidate_rejections": 0,
                        "free_sample_failures": 0,
                        "attempts_executed": 1,
                        "total_iterations_across_attempts": 25 + 12 * difficulty_index,
                        "total_generated_candidates_across_attempts": 100,
                        "total_accepted_nodes_across_attempts": 20,
                    })
        mock_comparison = {
            "gan_solve_rate_percent": 100.0,
            "classical_solve_rate_percent": 90.0,
            "mean_gan_runtime_seconds_all_cases": 0.3,
            "mean_classical_runtime_seconds_all_cases": 0.2,
            "mean_gan_path_cost_on_both_solved": 1.0,
            "mean_classical_path_cost_on_both_solved": 1.2,
            "both_solved": 9,
            "gan_only_solved": 1,
            "classical_only_solved": 0,
            "neither_solved": 0,
        }
        mock_final_d = [
            {
                "step": 1,
                "training_accuracy": 50.0,
                "validation_accuracy": 50.0,
                "training_loss": 0.693,
                "validation_loss": 0.693,
            },
            {
                "step": 150,
                "training_accuracy": 50.1,
                "validation_accuracy": 49.9,
                "training_loss": 0.692,
                "validation_loss": 0.694,
            },
        ]
        add_metric_figures(gallery, mock_metrics, 98.0)
        add_training_figures(gallery, mock_history, 98.0)
        add_rollout_figures(
            gallery, mock_rollouts, [{"placeholder": True}], mock_comparison
        )
        add_discriminator_figure(gallery, mock_final_d)
        png_count = len(list(gallery.figures.glob("*.png")))
        svg_count = len(list(gallery.figures.glob("*.svg")))
        csv_count = len(list(gallery.data.glob("*.csv")))
        if not (png_count == svg_count == csv_count == len(gallery.index)):
            raise RuntimeError(
                "Gallery full self-test found a figure/data count mismatch"
            )
        if png_count < 25:
            raise RuntimeError(
                f"Gallery full self-test produced too few figures: {png_count}"
            )
    print("RESULTS GALLERY SELF-TEST PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build numbered real-data figures and repackage V7.7 results."
    )
    parser.add_argument("--package_dir")
    parser.add_argument("--path_images_dir", default="")
    parser.add_argument(
        "--consume_path_images",
        action="store_true",
        help=(
            "After verified copying, remove only a package-local directory "
            "named .path_image_staging."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.package_dir:
        parser.error("--package_dir is required unless --self_test is used")

    started = time.perf_counter()
    package = Path(args.package_dir).expanduser().resolve()
    if not package.is_dir():
        raise FileNotFoundError(f"Final package directory not found: {package}")
    validate_required(package)
    gallery_root = package / "08_RESULTS_GALLERY"
    if gallery_root.exists() and any(gallery_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Results gallery already exists: {gallery_root}. Add --overwrite to rebuild it."
            )
        shutil.rmtree(gallery_root)
    gallery = Gallery(gallery_root)

    metrics = read_json(package / "FINAL_METRICS.json")
    verification = read_json(package / "verification_report.json")
    run_config = read_json(
        package / "05_MODEL_AND_DATASET_FILES" / "run_config.json"
    )
    history = read_csv(
        package / "05_MODEL_AND_DATASET_FILES" / "model" / "training_history.csv"
    )
    final_d_path = (
        package / "05_MODEL_AND_DATASET_FILES" / "model"
        / "final_discriminator_training_log.csv"
    )
    final_d_rows = read_csv(final_d_path) if final_d_path.is_file() else []
    benchmark_data = package / "04_OVERALL_BENCHMARK" / "02_OVERALL_BENCHMARK_DATA"
    rollout_rows = read_csv(benchmark_data / "all_rollout_cases.csv")
    classical_path = benchmark_data / "testing_classical_basic_rrt_cases.csv"
    comparison_path = benchmark_data / "testing_gan_vs_classical_paired_comparison.json"
    classical_rows = read_csv(classical_path) if classical_path.is_file() else []
    comparison = read_json(comparison_path) if comparison_path.is_file() else None

    tasks = [
        ("prediction metrics", lambda: add_metric_figures(
            gallery, metrics, float(run_config.get("target_accuracy", 98.0))
        )),
        ("training history", lambda: add_training_figures(
            gallery, history, float(run_config.get("target_accuracy", 98.0))
        )),
        ("rollout benchmarks", lambda: add_rollout_figures(
            gallery, rollout_rows, classical_rows, comparison
        )),
        ("discriminator audit", lambda: add_discriminator_figure(
            gallery, final_d_rows
        )),
    ]
    for index, (label, task) in enumerate(tasks, start=1):
        task_started = time.perf_counter()
        task()
        elapsed = time.perf_counter() - started
        eta = elapsed / index * (len(tasks) - index)
        print(
            f"GALLERY STAGE {index}/{len(tasks)}: {label} complete | "
            f"stage={format_duration(time.perf_counter() - task_started)} | "
            f"total={format_duration(elapsed)} | ETA={format_duration(eta)}",
            flush=True,
        )

    write_csv(gallery_root / "FIGURE_INDEX.csv", gallery.index)
    caption_lines = []
    for entry in gallery.index:
        caption_lines.extend([
            f"{entry['figure']}. {entry['title']}",
            entry["caption"],
            f"PNG: {entry['png']}",
            f"Data: {entry['data_csv']}",
            "",
        ])
    (gallery_root / "FIGURE_CAPTIONS.txt").write_text(
        "\n".join(caption_lines), encoding="utf-8"
    )
    write_csv(
        gallery.tables / "all_final_metrics_long_format.csv",
        flatten_final_metrics(metrics),
    )
    shutil.copy2(
        benchmark_data / "all_rollout_cases.csv",
        gallery.tables / "all_rollout_cases.csv",
    )
    if comparison_path.is_file():
        shutil.copy2(
            comparison_path,
            gallery.tables / "testing_gan_vs_classical_paired_comparison.json",
        )

    path_images = (
        Path(args.path_images_dir).expanduser().resolve()
        if args.path_images_dir
        else gallery_root / "04_PATH_IMAGES"
    )
    path_image_count = count_path_images(path_images)
    if path_images.is_dir() and path_images != gallery_root / "04_PATH_IMAGES":
        target = gallery_root / "04_PATH_IMAGES"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path_images, target)
        path_image_count = count_path_images(target)
        if args.consume_path_images:
            if (
                path_images.name != ".path_image_staging"
                or package not in path_images.parents
            ):
                raise ValueError(
                    "--consume_path_images only permits the explicit "
                    "package-local .path_image_staging directory"
                )
            shutil.rmtree(path_images)

    path_index_path = gallery_root / "04_PATH_IMAGES" / "rendered_case_index.csv"
    path_figure_rows = []
    if path_index_path.is_file():
        for index, row in enumerate(read_csv(path_index_path), start=1):
            path_figure_rows.append({
                "figure": f"Path Figure {index}",
                "split": row.get("split"),
                "difficulty": row.get("difficulty"),
                "row_index": row.get("row_index"),
                "solved": row.get("solved"),
                "reason": row.get("reason"),
                "direct_start_to_goal_route_blocked": row.get(
                    "direct_start_to_goal_route_blocked"
                ),
                "caption": (
                    f"Real {row.get('split')} {row.get('difficulty')} GAN-RRT "
                    f"rollout for dataset row {row.get('row_index')}; "
                    f"status={row.get('solved')} ({row.get('reason')})."
                ),
                "image": row.get("picture"),
            })
        write_csv(gallery_root / "PATH_FIGURE_INDEX.csv", path_figure_rows)

    write_text_summary(
        gallery_root / "00_RESULTS_SUMMARY.txt",
        metrics,
        verification,
        len(gallery.index),
        path_image_count,
    )
    (gallery_root / "README.txt").write_text(
        "V7.7 RESULTS GALLERY\n\n"
        "01_FIGURES contains numbered 300-DPI PNG and vector SVG figures.\n"
        "02_FIGURE_DATA contains the exact CSV values plotted in every figure.\n"
        "03_SUMMARY_TABLES contains long-format final metrics and raw rollout rows.\n"
        "04_PATH_IMAGES contains real rerun planner images when requested.\n\n"
        "No value or path in this directory is fabricated.\n",
        encoding="utf-8",
    )

    verification_rows = {
        "gallery_version": GALLERY_VERSION,
        "figures": len(gallery.index),
        "png_figures": len(list(gallery.figures.glob("*.png"))),
        "svg_figures": len(list(gallery.figures.glob("*.svg"))),
        "figure_data_csv_files": len(list(gallery.data.glob("*.csv"))),
        "path_images": path_image_count,
        "every_figure_has_data": bool(
            len(gallery.index)
            == len(list(gallery.figures.glob("*.png")))
            == len(list(gallery.figures.glob("*.svg")))
            == len(list(gallery.data.glob("*.csv")))
        ),
        "source_data": [
            "FINAL_METRICS.json",
            "training_history.csv",
            "final_discriminator_training_log.csv",
            "all_rollout_cases.csv",
            "testing_gan_vs_classical_paired_comparison.json",
        ],
        "fabricated_values": False,
    }
    if not verification_rows["every_figure_has_data"]:
        raise RuntimeError("Figure/data one-to-one verification failed")
    atomic_json(gallery_root / "GALLERY_VERIFICATION.json", verification_rows)

    print("REBUILDING VERIFIED PACKAGE ZIP", flush=True)
    zip_path, manifest = refresh_manifest_and_zip(package)
    total_elapsed = time.perf_counter() - started
    print(
        f"FINAL RESULTS GALLERY COMPLETE | figures={len(gallery.index)} | "
        f"path images={path_image_count} | files verified={len(manifest)} | "
        f"elapsed={format_duration(total_elapsed)}",
        flush=True,
    )
    print(f"FINAL FOLDER: {package}", flush=True)
    print(f"FINAL ZIP: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
