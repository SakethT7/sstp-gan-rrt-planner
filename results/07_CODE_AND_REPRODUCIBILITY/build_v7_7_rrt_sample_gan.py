#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass
import argparse
import csv
import json
import math
import os
import shutil
import shlex
import time
import hashlib
import sys
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HOME = Path.home()
SCRIPT_VERSION = "7.7.0-invariant-constraint-attention-control-gan"
# V7.7 deliberately starts from scratch. V7.5/V7.6 checkpoints contain a
# large absolute-coordinate trunk that achieved 99.9% training accuracy while
# remaining near 77% on held-out environments. Importing those weights would
# reintroduce the exact shortcut this version removes.
WARM_START_SCRIPT_VERSIONS = {SCRIPT_VERSION}
CONDITION_SCHEMA = (
    "[parent_x,parent_y,parent_vx,parent_vy,sample_x,sample_y,"
    "absolute_obstacle_x_y_r...]"
)
GENERATOR_OUTPUT_SCHEMA = "[delta_x,delta_y,next_vx,next_vy]"
METHOD_NAME = "v7_7_invariant_constraint_attention_control_consistent_gan_steering"

# The strict V7.3 expert uses the fixed discrete-time LQR gains documented in
# generate_v7_rrt_sample_dataset.py.  Expressed in normalized one-step velocity
# increment units, these constants provide a label-free nominal-control feature
# to V7.7's invariant residual model. They do not replace the learned output and
# they do not use validation/testing information.
NOMINAL_LQR_POSITION_INCREMENT_GAIN = 4.394738642272157
NOMINAL_LQR_VELOCITY_INCREMENT_GAIN = 0.41360283074019036
NOMINAL_LQR_STEP_DISTANCE_NORMALIZED = 0.14

# Fixed teacher contract. These values are verified against dataset metadata
# before a model is created. Keeping them named in one place prevents the
# generator features and the exact reconstruction diagnostic from drifting.
PHYSICAL_POSITION_SCALE = 50.0
PHYSICAL_VELOCITY_SCALE = 6.0
PHYSICAL_TIME_STEP = 0.12
PHYSICAL_MAX_ACCELERATION = 5.0
PHYSICAL_CONFIGURATION_RADIUS = 0.8
CBF_K0 = 2.0
CBF_K1 = 2.8


def format_duration(seconds):
    """Format a measured or estimated duration without false precision."""
    seconds = max(0.0, float(seconds))
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{seconds:.1f}s"


def measured_progress(label, completed, total, started_at):
    """Return an empirical elapsed/ETA line based only on completed work."""
    completed = int(completed)
    total = int(total)
    elapsed = max(0.0, time.perf_counter() - float(started_at))
    if completed <= 0:
        eta = None
    else:
        eta = elapsed / completed * max(0, total - completed)
    eta_text = "estimating" if eta is None else format_duration(eta)
    percent = 100.0 * completed / max(1, total)
    return (
        f"{label}: {completed}/{total} ({percent:.2f}%) | "
        f"elapsed={format_duration(elapsed)} | ETA={eta_text}"
    )


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    path = Path(path)
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def write_csv(path, rows):
    path = Path(path)
    mkdir(path.parent)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(obj, path):
    path = Path(path).expanduser()
    mkdir(path.parent)
    temporary = Path(str(path) + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions that predate weights_only.
        return torch.load(path, map_location=map_location)


def validate_warm_start_checkpoint(checkpoint, args, cond_dim):
    """Validate a same-architecture V7.7 checkpoint before importing it.

    A warm start is intentionally different from ``--resume``: it imports only
    the validation-selected model into a new experiment. Optimizer state,
    completed-epoch counters, and testing information are never imported.
    """
    if checkpoint.get("format") != "v7_rrt_sample_gan_training_checkpoint_v1":
        raise RuntimeError("Warm-start checkpoint has an unsupported format")
    source_version = checkpoint.get("script_version")
    if source_version not in WARM_START_SCRIPT_VERSIONS:
        raise RuntimeError(
            "Warm-start checkpoint was not produced by an approved V7.7 "
            f"trainer: {source_version!r}"
        )
    if checkpoint.get("dataset_sha256") != args.dataset_sha256:
        raise RuntimeError(
            "Warm-start checkpoint belongs to a different dataset; refusing "
            "cross-dataset fine-tuning"
        )
    if int(checkpoint.get("cond_dim", -1)) != int(cond_dim):
        raise RuntimeError("Warm-start checkpoint condition dimension does not match")
    if int(checkpoint.get("noise_dim", -1)) != int(args.noise_dim):
        raise RuntimeError("Warm-start checkpoint noise dimension does not match")
    if checkpoint.get("best_generator_state_dict") is None:
        raise RuntimeError("Warm-start checkpoint does not contain a selected generator")
    if checkpoint.get("best_discriminator_state_dict") is None:
        raise RuntimeError("Warm-start checkpoint does not contain a selected discriminator")
    history = list(checkpoint.get("history", []))
    if not history:
        raise RuntimeError("Warm-start checkpoint does not contain validation history")
    if not all("selection_score" in row for row in history):
        raise RuntimeError("Warm-start validation history is incomplete")
    selected = max(history, key=lambda row: float(row["selection_score"]))
    if not math.isclose(
        float(selected["selection_score"]),
        float(checkpoint.get("best_score", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "Warm-start checkpoint best score disagrees with its validation history"
        )
    return selected


def training_checkpoint_path(args):
    return Path(str(Path(args.output_dir).expanduser()) + "_TRAINING_CHECKPOINT.pt")


def final_discriminator_checkpoint_path(args):
    return Path(str(Path(args.output_dir).expanduser()) + "_FINAL_DISCRIMINATOR_CHECKPOINT.pt")


def checkpoint_run_config(args):
    """Capture every user choice that can change a resumed run's meaning."""
    excluded = {
        "resume", "preflight_only", "dataset_sha256", "max_turn_rad",
        "script_version", "restart_training", "overwrite_output",
        "stop_after_training", "allow_below_target_finalization",
    }
    return {
        key: value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def model_state_sha256(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(str(array.shape).encode("ascii"))
        h.update(array.tobytes())
    return h.hexdigest()


def load_dataset(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise RuntimeError(f"Dataset path is not a regular file: {path}")
    keys = [
        "train_X", "train_y", "validation_X", "validation_y",
        "testing_X", "testing_y", "train_sample_xy",
        "validation_sample_xy", "testing_sample_xy", "training_goal_state",
        "validation_goal_state", "testing_goal_state", "target_source",
        "obstacle_coordinate_schema", "train_environment_id",
        "validation_environment_id", "testing_environment_id",
        "train_scenario_id", "validation_scenario_id", "testing_scenario_id",
        "condition_schema", "generator_script_version", "basic_rrt_contract",
        "collision_margin", "workspace_min", "workspace_max", "nominal_dt",
        "position_scale", "velocity_scale", "goal_radius",
        "continuous_edge_safety_schema", "supervision_lookahead_steps",
        "geometric_connectivity_filter",
        "cbf_control_filter_schema",
    ]
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in keys if k not in data.files]
        if missing:
            raise RuntimeError(
                "Dataset is not a strict V7 sample-conditioned dataset. Missing arrays: "
                f"{missing}"
            )

        def text_scalar(key):
            value = np.asarray(data[key]).reshape(-1)[0]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        out = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in (
                "train_X", "train_y", "validation_X", "validation_y",
                "testing_X", "testing_y", "train_sample_xy",
                "validation_sample_xy", "testing_sample_xy",
                "training_goal_state", "validation_goal_state", "testing_goal_state",
            )
        }
        for key in (
            "train_environment_id", "validation_environment_id",
            "testing_environment_id", "train_scenario_id",
            "validation_scenario_id", "testing_scenario_id",
        ):
            out[key] = np.asarray(data[key], dtype=np.int64)
        out["target_source"] = np.asarray([
            v.decode("utf-8") if isinstance(v, bytes) else str(v)
            for v in np.asarray(data["target_source"]).reshape(-1).tolist()
        ])
        out["obstacle_coordinate_schema"] = text_scalar(
            "obstacle_coordinate_schema"
        )
        out["condition_schema"] = (
            text_scalar("condition_schema")
            if "condition_schema" in data.files else CONDITION_SCHEMA
        )
        out["generator_script_version"] = (
            text_scalar("generator_script_version")
        )
        out["basic_rrt_contract"] = text_scalar("basic_rrt_contract")
        out["continuous_edge_safety_schema"] = text_scalar(
            "continuous_edge_safety_schema"
        )
        out["geometric_connectivity_filter"] = text_scalar(
            "geometric_connectivity_filter"
        )
        out["cbf_control_filter_schema"] = text_scalar(
            "cbf_control_filter_schema"
        )
        out["collision_margin"] = float(
            np.asarray(data["collision_margin"]).reshape(-1)[0]
        ) if "collision_margin" in data.files else 0.0
        out["workspace_min"] = float(
            np.asarray(data["workspace_min"]).reshape(-1)[0]
        ) if "workspace_min" in data.files else 0.0
        out["workspace_max"] = float(
            np.asarray(data["workspace_max"]).reshape(-1)[0]
        ) if "workspace_max" in data.files else 1.0
        out["nominal_dt"] = float(
            np.asarray(data["nominal_dt"]).reshape(-1)[0]
        )
        out["position_scale"] = np.asarray(
            data["position_scale"], dtype=np.float32
        ).reshape(-1)
        out["velocity_scale"] = float(
            np.asarray(data["velocity_scale"]).reshape(-1)[0]
        )
        out["goal_radius"] = float(
            np.asarray(data["goal_radius"]).reshape(-1)[0]
        )
        out["supervision_lookahead_steps"] = int(
            np.asarray(data["supervision_lookahead_steps"]).reshape(-1)[0]
        )

    if out["target_source"].tolist() != ["recorded_rrt_sample_xy"] * 3:
        raise RuntimeError(
            "V7 refuses legacy final-goal supervision. Regenerate the dataset with "
            "the strict sample/nearest-parent/accepted-child generator."
        )
    expected_generator_version = "7.3.0-strict-one-step-sample-parent-child-cbf-qp"
    if out["generator_script_version"] != expected_generator_version:
        raise RuntimeError(
            f"Dataset generator version {out['generator_script_version']!r} does "
            f"not match required {expected_generator_version!r}. Regenerate the "
            "dataset; older versions lack the final CBF-QP, continuous-edge, "
            "and local-horizon fixes."
        )
    expected_rrt_contract = (
        "sample->nearest_parent->lqr_cbf_steer->safety_check->one_child"
    )
    if out["basic_rrt_contract"] != expected_rrt_contract:
        raise RuntimeError(
            f"Dataset RRT contract {out['basic_rrt_contract']!r} does not match "
            f"{expected_rrt_contract!r}"
        )
    expected_edge_safety_schema = (
        "constant_acceleration_curve_bound_plus_swept_chord"
    )
    if out["continuous_edge_safety_schema"] != expected_edge_safety_schema:
        raise RuntimeError(
            "Unsupported continuous edge-safety schema: "
            f"{out['continuous_edge_safety_schema']!r}; expected "
            f"{expected_edge_safety_schema!r}"
        )
    expected_connectivity_filter = (
        "four_neighbor_configuration_space_grid_same_component"
    )
    if out["geometric_connectivity_filter"] != expected_connectivity_filter:
        raise RuntimeError(
            f"Unsupported geometric connectivity filter: "
            f"{out['geometric_connectivity_filter']!r}; expected "
            f"{expected_connectivity_filter!r}"
        )
    expected_cbf_filter = (
        "exact_2d_active_set_projection_of_lqr_onto_hocbf_halfspaces"
    )
    if out["cbf_control_filter_schema"] != expected_cbf_filter:
        raise RuntimeError(
            f"Unsupported CBF control filter: {out['cbf_control_filter_schema']!r}; "
            f"expected {expected_cbf_filter!r}"
        )
    expected_obstacle_schema = (
        "absolute_normalized_center_x_center_y_radius_nearest_first"
    )
    if out["obstacle_coordinate_schema"] != expected_obstacle_schema:
        raise RuntimeError(
            "Unsupported obstacle schema: "
            f"{out['obstacle_coordinate_schema']!r}; expected {expected_obstacle_schema!r}"
        )
    if out["position_scale"].shape != (2,) or np.any(out["position_scale"] <= 0):
        raise RuntimeError("Dataset position_scale must contain two positive values")
    if out["velocity_scale"] <= 0 or out["nominal_dt"] <= 0:
        raise RuntimeError("Dataset velocity_scale and nominal_dt must be positive")
    if not 0.0 < out["goal_radius"] < 0.5:
        raise RuntimeError("Dataset goal_radius must be in normalized range (0, 0.5)")
    if out["supervision_lookahead_steps"] < 1:
        raise RuntimeError("Dataset supervision_lookahead_steps must be positive")
    if out["supervision_lookahead_steps"] != 1:
        raise RuntimeError(
            "V7.7 requires one-step expert labels; regenerate with --lookahead_steps 1"
        )
    if not math.isclose(
        float(out["position_scale"][0]), float(out["position_scale"][1]),
        rel_tol=0.0, abs_tol=1e-6,
    ):
        raise RuntimeError("V7 circular-obstacle evaluation requires equal x/y position scales")

    feature_dims = set()
    for split, xkey, ykey, sample_key, goal_key, env_key, scenario_key in [
        (
            "training", "train_X", "train_y", "train_sample_xy",
            "training_goal_state", "train_environment_id", "train_scenario_id",
        ),
        (
            "validation", "validation_X", "validation_y", "validation_sample_xy",
            "validation_goal_state", "validation_environment_id", "validation_scenario_id",
        ),
        (
            "testing", "testing_X", "testing_y", "testing_sample_xy",
            "testing_goal_state", "testing_environment_id", "testing_scenario_id",
        ),
    ]:
        X = out[xkey]
        y = out[ykey]
        sample_xy = out[sample_key]
        goal_state = out[goal_key]
        env_id = out[env_key]
        scenario_id = out[scenario_key]
        if X.ndim != 2 or y.ndim != 2:
            raise RuntimeError(f"{split}: X and y must be 2D arrays")
        if len(X) == 0 or len(y) == 0:
            raise RuntimeError(f"{split}: X and y must not be empty")
        row_lengths = [len(X), len(y), len(sample_xy), len(goal_state), len(env_id), len(scenario_id)]
        if len(set(row_lengths)) != 1:
            raise RuntimeError(f"{split}: row-count mismatch: {row_lengths}")
        if X.shape[1] < 6 or (X.shape[1] - 6) % 3 != 0:
            raise RuntimeError(f"{split}: invalid strict condition shape {X.shape}")
        if y.shape[1] != 4:
            raise RuntimeError(f"{split}: y must be full state [x,y,vx,vy], got {y.shape}")
        if sample_xy.shape != (len(X), 2) or goal_state.shape != (len(X), 4):
            raise RuntimeError(f"{split}: invalid sample or final-goal shape")
        if not all(np.isfinite(a).all() for a in (X, y, sample_xy, goal_state)):
            raise RuntimeError(f"{split}: dataset contains NaN or infinity")
        if not np.array_equal(X[:, 4:6], sample_xy):
            raise RuntimeError(f"{split}: X sample columns do not exactly match saved RRT samples")
        if np.any(X[:, 0:2] < 0.0) or np.any(X[:, 0:2] > 1.0):
            raise RuntimeError(f"{split}: parent position lies outside normalized workspace")
        if np.any(sample_xy < 0.0) or np.any(sample_xy > 1.0):
            raise RuntimeError(f"{split}: sampled target lies outside normalized workspace")
        if np.any(y[:, 0:2] < 0.0) or np.any(y[:, 0:2] > 1.0):
            raise RuntimeError(f"{split}: expert child lies outside normalized workspace")
        parent_speed = np.linalg.norm(X[:, 2:4], axis=1)
        child_speed = np.linalg.norm(y[:, 2:4], axis=1)
        if np.any(parent_speed > 1.0 + 2e-5):
            raise RuntimeError(
                f"{split}: normalized parent speed exceeds the recorded physical limit"
            )
        if np.any(child_speed > 1.0 + 2e-5):
            raise RuntimeError(
                f"{split}: normalized child speed exceeds the recorded physical limit"
            )

        triples = X[:, 6:].reshape(len(X), -1, 3)
        centers = triples[:, :, 0:2]
        radii = triples[:, :, 2]
        active = radii > 0.0
        if np.any(radii < 0.0) or np.any(radii > 0.5):
            raise RuntimeError(f"{split}: invalid normalized obstacle radius")
        if np.any(centers[active] < 0.0) or np.any(centers[active] > 1.0):
            raise RuntimeError(f"{split}: active obstacle center is not absolute normalized XY")
        if np.any(np.abs(centers[~active]) > 1e-8):
            raise RuntimeError(f"{split}: padded obstacle triples must be all zero")
        # Nearest-first ordering is part of the schema and is reconstructed at
        # every rollout node, so verify it before training.
        for row_i in range(len(X)):
            row_active = active[row_i]
            if not np.any(row_active):
                continue
            clearance_order = np.linalg.norm(
                centers[row_i, row_active] - X[row_i, 0:2], axis=1
            ) - radii[row_i, row_active]
            if np.any(np.diff(clearance_order) < -1e-6):
                raise RuntimeError(f"{split}: obstacle triples are not nearest-first at row {row_i}")
        feature_dims.add(int(X.shape[1]))

    if len(feature_dims) != 1:
        raise RuntimeError(f"All splits must use the same X feature count, got {sorted(feature_dims)}")

    env_sets = [
        set(out[key].tolist()) for key in (
            "train_environment_id", "validation_environment_id", "testing_environment_id"
        )
    ]
    if env_sets[0] & env_sets[1] or env_sets[0] & env_sets[2] or env_sets[1] & env_sets[2]:
        raise RuntimeError("Environment leakage detected across dataset splits")
    scenario_sets = [
        set(out[key].tolist()) for key in (
            "train_scenario_id", "validation_scenario_id", "testing_scenario_id"
        )
    ]
    if (
        scenario_sets[0] & scenario_sets[1]
        or scenario_sets[0] & scenario_sets[2]
        or scenario_sets[1] & scenario_sets[2]
    ):
        raise RuntimeError("Scenario leakage detected across dataset splits")
    return out


def validate_nominal_lqr_feature_contract(data):
    """Refuse to apply V7.7's fixed physics features to incompatible data."""
    expected_position_scale = np.asarray(
        [PHYSICAL_POSITION_SCALE, PHYSICAL_POSITION_SCALE], dtype=np.float32
    )
    if not np.allclose(
        data["position_scale"], expected_position_scale, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError(
            "V7.7 physics features require position_scale=[50,50]"
        )
    if not math.isclose(
        float(data["velocity_scale"]), PHYSICAL_VELOCITY_SCALE,
        rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError("V7.7 physics features require velocity_scale=6")
    if not math.isclose(
        float(data["nominal_dt"]), PHYSICAL_TIME_STEP,
        rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError("V7.7 physics features require nominal_dt=0.12")
    expected_step = 7.0 / 50.0
    if not math.isclose(
        NOMINAL_LQR_STEP_DISTANCE_NORMALIZED,
        expected_step,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Internal V7.7 normalized LQR step constant is invalid")
    print(
        "NOMINAL-LQR FEATURE CONTRACT PASSED: fixed V7.3 expert metadata and "
        "normalized gains agree",
        flush=True,
    )


def condition_only_expert_reconstruction_diagnostics(data, args):
    """Recreate the documented teacher from encoded conditions only.

    This read-only ceiling check is never used as a generator prediction or a
    training target. Only training and validation are inspected; testing stays
    untouched until final evaluation.
    """
    position_scale = float(data["position_scale"][0])
    velocity_scale = float(data["velocity_scale"])
    dt = float(data["nominal_dt"])
    configuration_radius = float(data["collision_margin"] * position_scale)
    max_acceleration = PHYSICAL_MAX_ACCELERATION
    cbf_k0 = CBF_K0
    cbf_k1 = CBF_K1
    position_gain = (
        NOMINAL_LQR_POSITION_INCREMENT_GAIN
        * velocity_scale
        / (position_scale * dt)
    )
    velocity_gain = NOMINAL_LQR_VELOCITY_INCREMENT_GAIN / dt
    physical_step = NOMINAL_LQR_STEP_DISTANCE_NORMALIZED * position_scale

    def project_control(position, velocity, nominal, obstacles):
        rows = []
        rhs = []
        for center_x, center_y, radius in obstacles:
            if radius <= 0.0:
                continue
            relative = position - np.asarray([center_x, center_y])
            combined_radius = float(radius + configuration_radius)
            barrier = float(relative @ relative - combined_radius ** 2)
            barrier_dot = float(2.0 * relative @ velocity)
            rows.append(2.0 * relative)
            rhs.append(
                float(
                    -2.0 * (velocity @ velocity)
                    - cbf_k1 * barrier_dot
                    - cbf_k0 * barrier
                )
            )
        width = position_scale
        rows.extend(
            [
                np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
                np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
                np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]),
                np.asarray([0.0, 1.0]), np.asarray([0.0, -1.0]),
            ]
        )
        rhs.extend(
            [
                float(-cbf_k1 * velocity[0] - cbf_k0 * (position[0] - configuration_radius)),
                float(cbf_k1 * velocity[0] - cbf_k0 * (width - configuration_radius - position[0])),
                float(-cbf_k1 * velocity[1] - cbf_k0 * (position[1] - configuration_radius)),
                float(cbf_k1 * velocity[1] - cbf_k0 * (width - configuration_radius - position[1])),
                -max_acceleration, -max_acceleration,
                -max_acceleration, -max_acceleration,
            ]
        )
        matrix = np.asarray(rows, dtype=np.float64)
        boundary = np.asarray(rhs, dtype=np.float64)

        def feasible(candidate):
            return bool(np.all(matrix @ candidate >= boundary - 1e-8))

        clipped = np.clip(nominal, -max_acceleration, max_acceleration)
        if feasible(clipped):
            return clipped
        best = None
        best_distance = float("inf")

        def consider(candidate):
            nonlocal best, best_distance
            if not np.isfinite(candidate).all() or not feasible(candidate):
                return
            distance = float(np.sum((candidate - nominal) ** 2))
            if distance < best_distance:
                best = candidate.copy()
                best_distance = distance

        for index, normal in enumerate(matrix):
            denominator = float(normal @ normal)
            if denominator > 1e-14:
                consider(
                    nominal
                    + (boundary[index] - float(normal @ nominal))
                    * normal
                    / denominator
                )
        for first in range(len(matrix)):
            for second in range(first + 1, len(matrix)):
                active = np.stack([matrix[first], matrix[second]], axis=0)
                if abs(float(np.linalg.det(active))) <= 1e-12:
                    continue
                consider(
                    np.linalg.solve(
                        active,
                        np.asarray([boundary[first], boundary[second]]),
                    )
                )
        return best

    report = {}
    for split, x_key, y_key in (
        ("training", "train_X", "train_y"),
        ("validation", "validation_X", "validation_y"),
    ):
        conditions = np.asarray(data[x_key], dtype=np.float64)
        labels = np.asarray(data[y_key], dtype=np.float64)
        predicted = np.full((len(conditions), 4), np.nan, dtype=np.float64)
        failed = 0
        for index, condition in enumerate(conditions):
            position = condition[0:2] * position_scale
            velocity = condition[2:4] * velocity_scale
            target_delta = condition[4:6] * position_scale - position
            target_distance = float(np.linalg.norm(target_delta))
            if target_distance > physical_step:
                target_delta *= physical_step / max(target_distance, 1e-12)
            nominal = position_gain * target_delta - velocity_gain * velocity
            obstacles = condition[6:].reshape(-1, 3).copy()
            obstacles[:, 0:2] *= position_scale
            obstacles[:, 2] *= position_scale
            control = project_control(position, velocity, nominal, obstacles)
            if control is None:
                failed += 1
                continue
            next_position = position + velocity * dt + 0.5 * control * dt * dt
            next_velocity = velocity + control * dt
            speed = float(np.linalg.norm(next_velocity))
            if speed > velocity_scale:
                next_velocity *= velocity_scale / max(speed, 1e-12)
            predicted[index, 0:2] = next_position / position_scale
            predicted[index, 2:4] = next_velocity / velocity_scale

        finite = np.isfinite(predicted).all(axis=1)
        position_error = np.full(len(labels), np.inf, dtype=np.float64)
        velocity_error = np.full(len(labels), np.inf, dtype=np.float64)
        state_error = np.full(len(labels), np.inf, dtype=np.float64)
        position_error[finite] = np.linalg.norm(
            predicted[finite, 0:2] - labels[finite, 0:2], axis=1
        )
        velocity_error[finite] = np.linalg.norm(
            predicted[finite, 2:4] - labels[finite, 2:4], axis=1
        )
        state_error[finite] = np.linalg.norm(
            predicted[finite] - labels[finite], axis=1
        )
        finite_error = state_error[np.isfinite(state_error)]
        report[split] = {
            "rows": int(len(conditions)),
            "failed_qp_reconstructions": int(failed),
            "condition_only_position_accuracy_percent": float(
                100.0 * np.mean(position_error <= args.position_accuracy_threshold)
            ),
            "condition_only_velocity_accuracy_percent": float(
                100.0 * np.mean(velocity_error <= args.velocity_accuracy_threshold)
            ),
            "condition_only_full_state_accuracy_percent": float(
                100.0 * np.mean(state_error <= args.state_accuracy_threshold)
            ),
            "condition_only_mean_full_state_error": float(
                np.mean(finite_error)
            ) if len(finite_error) else None,
            "condition_only_p95_full_state_error": float(
                np.percentile(finite_error, 95)
            ) if len(finite_error) else None,
            "condition_only_max_full_state_error": float(
                np.max(finite_error)
            ) if len(finite_error) else None,
        }
    return report


def target_condition_diagnostics(data, include_testing=True):
    report = {}
    split_specs = [
        ("training", "train_X", "train_y"),
        ("validation", "validation_X", "validation_y"),
    ]
    if include_testing:
        split_specs.append(("testing", "testing_X", "testing_y"))
    for split, xkey, ykey in split_specs:
        X = data[xkey]
        y = data[ykey]
        before = np.linalg.norm(X[:, 4:6] - X[:, 0:2], axis=1)
        after = np.linalg.norm(X[:, 4:6] - y[:, 0:2], axis=1)
        progress = before - after
        target = X[:, 4:6]
        report[split] = {
            "rows": int(len(X)),
            "condition_schema": CONDITION_SCHEMA,
            "positive_progress_percent": float(100.0 * np.mean(progress > 0.0)),
            "nonnegative_progress_percent": float(100.0 * np.mean(progress >= -1e-8)),
            "mean_expert_progress_to_conditioned_target": float(np.mean(progress)),
            "p05_expert_progress_to_conditioned_target": float(np.percentile(progress, 5)),
            "target_x_min": float(np.min(target[:, 0])),
            "target_x_max": float(np.max(target[:, 0])),
            "target_y_min": float(np.min(target[:, 1])),
            "target_y_max": float(np.max(target[:, 1])),
        }
    return report


def physical_scenario_signature(cond, goal_state, decimals=4):
    """Identify goal/obstacle geometry without using transition state or velocity."""
    canonical_obstacles = sorted(
        parse_obstacles(cond), key=lambda item: (item[0], item[1], item[2])
    )
    fields = np.concatenate([
        np.asarray(goal_state, dtype=np.float32)[0:2],
        np.asarray(canonical_obstacles, dtype=np.float32).reshape(-1),
    ])
    return np.ascontiguousarray(np.round(fields, decimals=decimals)).tobytes()


def split_scenario_overlap_diagnostics(data, decimals=4):
    """Use generator-recorded scenario IDs; geometry ordering cannot hide leakage."""
    del decimals
    signatures = {}
    specs = [
        ("training", "train_X", "train_scenario_id", "train_environment_id"),
        (
            "validation", "validation_X", "validation_scenario_id",
            "validation_environment_id",
        ),
        ("testing", "testing_X", "testing_scenario_id", "testing_environment_id"),
    ]
    environment_sets = {}
    for split, xkey, scenario_key, environment_key in specs:
        signatures[split] = set(data[scenario_key].tolist())
        environment_sets[split] = set(data[environment_key].tolist())
        signatures[f"{split}_row_count"] = len(data[xkey])
    pairs = {}
    for left, right in (("training", "validation"), ("training", "testing"), ("validation", "testing")):
        overlap = signatures[left] & signatures[right]
        environment_overlap = environment_sets[left] & environment_sets[right]
        pairs[f"{left}_vs_{right}"] = {
            "overlapping_scenario_ids": int(len(overlap)),
            "overlapping_environment_ids": int(len(environment_overlap)),
            "left_unique_scenario_ids": int(len(signatures[left])),
            "right_unique_scenario_ids": int(len(signatures[right])),
            "left_rows": int(signatures[f"{left}_row_count"]),
            "right_rows": int(signatures[f"{right}_row_count"]),
        }
    return {
        "identity_source": "generator-recorded scenario_id and environment_id",
        "pairs": pairs,
        "any_overlap": bool(any(
            v["overlapping_scenario_ids"] > 0
            or v["overlapping_environment_ids"] > 0
            for v in pairs.values()
        )),
        "interpretation": (
            "Scenario overlap can make validation/testing results optimistic. "
            "V7 assigns whole environments to exactly one split."
        ),
    }


def target_direction_step_speed(X, y):
    """Return the exact local motion label [dx,dy,next_vx,next_vy].

    The historical function name is retained so old internal call sites remain
    checkpoint-safe, but V7 does not force velocity to align with displacement.
    This is a lossless representation of the full expert state relative to the
    selected parent.
    """
    dpos = y[:, 0:2] - X[:, 0:2]
    return np.concatenate([dpos, y[:, 2:4]], axis=1).astype(np.float32)


def motion_params_in_sample_frame(X, params):
    """Rotate [delta_xy,next_velocity_xy] into the conditioned sample frame.

    This is an invertible, training-split-only coordinate transform. It gives
    the isotropic LQR mapping the same representation after any global rotation,
    which improves held-out generalization without changing labels or metrics.
    """
    to_sample = X[:, 4:6] - X[:, 0:2]
    distance = np.linalg.norm(to_sample, axis=1, keepdims=True)
    unit = to_sample / np.maximum(distance, 1e-8)
    zero_distance = distance[:, 0] <= 1e-8
    if np.any(zero_distance):
        unit[zero_distance] = np.asarray([1.0, 0.0], dtype=np.float32)
    perpendicular = np.stack([-unit[:, 1], unit[:, 0]], axis=1)
    delta = params[:, 0:2]
    velocity = params[:, 2:4]
    return np.stack(
        [
            np.sum(delta * unit, axis=1),
            np.sum(delta * perpendicular, axis=1),
            np.sum(velocity * unit, axis=1),
            np.sum(velocity * perpendicular, axis=1),
        ],
        axis=1,
    ).astype(np.float32)


D4_MATRICES = np.asarray(
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, -1.0], [1.0, 0.0]],
        [[-1.0, 0.0], [0.0, -1.0]],
        [[0.0, 1.0], [-1.0, 0.0]],
        [[-1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, -1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0], [-1.0, 0.0]],
    ],
    dtype=np.float32,
)


def augment_d4_batch(X, y, rng, probability=1.0, transform_ids=None):
    """Apply exact square symmetries to a training batch.

    The square workspace, circular obstacles, isotropic double-integrator
    dynamics, and componentwise symmetric acceleration bounds are invariant to
    all eight D4 rotations/reflections. Padding obstacle triples remain exactly
    zero. This function is training-only and never touches validation/testing.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] < 6 or (X.shape[1] - 6) % 3 != 0:
        raise ValueError("D4 augmentation received an invalid condition array")
    if y.ndim != 2 or y.shape != (len(X), 4):
        raise ValueError("D4 augmentation received an invalid state-label array")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("D4 augmentation probability must be between 0 and 1")
    if transform_ids is None:
        transform_ids = rng.integers(0, len(D4_MATRICES), size=len(X))
        if probability < 1.0:
            enabled = rng.random(len(X)) < probability
            transform_ids = np.where(enabled, transform_ids, 0)
    else:
        transform_ids = np.asarray(transform_ids, dtype=np.int64)
        if transform_ids.shape != (len(X),):
            raise ValueError("D4 transform_ids must contain one ID per row")
    if np.any(transform_ids < 0) or np.any(transform_ids >= len(D4_MATRICES)):
        raise ValueError("D4 transform ID is outside [0,7]")

    transformed_X = X.copy()
    transformed_y = y.copy()
    center = np.asarray([0.5, 0.5], dtype=np.float32)
    obstacle_count = (X.shape[1] - 6) // 3

    for transform_id in np.unique(transform_ids):
        rows = np.flatnonzero(transform_ids == transform_id)
        matrix = D4_MATRICES[int(transform_id)]
        if int(transform_id) == 0:
            continue
        for position_slice in (slice(0, 2), slice(4, 6)):
            points = X[rows, position_slice]
            transformed_X[rows, position_slice] = (
                (points - center) @ matrix.T + center
            )
        transformed_X[rows, 2:4] = X[rows, 2:4] @ matrix.T
        transformed_y[rows, 0:2] = (
            (y[rows, 0:2] - center) @ matrix.T + center
        )
        transformed_y[rows, 2:4] = y[rows, 2:4] @ matrix.T

        if obstacle_count:
            source_obstacles = X[rows, 6:].reshape(len(rows), obstacle_count, 3)
            target_obstacles = transformed_X[rows, 6:].reshape(
                len(rows), obstacle_count, 3
            )
            active = source_obstacles[:, :, 2] > 0.0
            transformed_centers = (
                (source_obstacles[:, :, 0:2] - center) @ matrix.T + center
            )
            target_obstacles[:, :, 0:2] = np.where(
                active[:, :, None], transformed_centers, 0.0
            )
            target_obstacles[:, :, 2] = source_obstacles[:, :, 2]
            transformed_X[rows, 6:] = target_obstacles.reshape(len(rows), -1)

    # Integer D4 transforms should preserve bounds exactly; clipping only
    # removes harmless float32 endpoint roundoff.
    transformed_X[:, 0:2] = np.clip(transformed_X[:, 0:2], 0.0, 1.0)
    transformed_X[:, 4:6] = np.clip(transformed_X[:, 4:6], 0.0, 1.0)
    transformed_y[:, 0:2] = np.clip(transformed_y[:, 0:2], 0.0, 1.0)
    return (
        np.ascontiguousarray(transformed_X, dtype=np.float32),
        np.ascontiguousarray(transformed_y, dtype=np.float32),
    )


def d4_augmentation_diagnostics(data, max_rows=512):
    """Numerically prove that the implemented augmentation preserves geometry."""
    gram = D4_MATRICES @ np.transpose(D4_MATRICES, (0, 2, 1))
    identity = np.eye(2, dtype=np.float32)[None, :, :]
    orthogonality_error = float(np.max(np.abs(gram - identity)))
    determinants = np.linalg.det(D4_MATRICES)
    if orthogonality_error > 1e-7 or not np.allclose(
        np.abs(determinants), 1.0, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("Internal D4 matrices are not orthogonal symmetries")
    if len(np.unique(D4_MATRICES.reshape(len(D4_MATRICES), -1), axis=0)) != 8:
        raise RuntimeError("Internal D4 matrix list contains duplicates")

    X = np.asarray(data["train_X"][:max_rows], dtype=np.float32)
    y = np.asarray(data["train_y"][:max_rows, 0:4], dtype=np.float32)
    if not len(X):
        raise RuntimeError("D4 diagnostics require a nonempty training split")
    original_sample_distance = np.linalg.norm(X[:, 4:6] - X[:, 0:2], axis=1)
    original_parent_speed = np.linalg.norm(X[:, 2:4], axis=1)
    original_child_speed = np.linalg.norm(y[:, 2:4], axis=1)
    original_delta = y[:, 0:2] - X[:, 0:2]
    obstacle_count = (X.shape[1] - 6) // 3
    max_distance_error = 0.0
    max_speed_error = 0.0
    max_vector_error = 0.0
    max_round_trip_error = 0.0
    padding_preserved = True

    rng = np.random.default_rng(0)
    for transform_id, matrix in enumerate(D4_MATRICES):
        ids = np.full(len(X), transform_id, dtype=np.int64)
        transformed_X, transformed_y = augment_d4_batch(
            X, y, rng, transform_ids=ids
        )
        max_distance_error = max(
            max_distance_error,
            float(np.max(np.abs(
                np.linalg.norm(
                    transformed_X[:, 4:6] - transformed_X[:, 0:2], axis=1
                ) - original_sample_distance
            ))),
        )
        max_speed_error = max(
            max_speed_error,
            float(np.max(np.abs(
                np.linalg.norm(transformed_X[:, 2:4], axis=1)
                - original_parent_speed
            ))),
            float(np.max(np.abs(
                np.linalg.norm(transformed_y[:, 2:4], axis=1)
                - original_child_speed
            ))),
        )
        expected_delta = original_delta @ matrix.T
        actual_delta = transformed_y[:, 0:2] - transformed_X[:, 0:2]
        max_vector_error = max(
            max_vector_error,
            float(np.max(np.abs(actual_delta - expected_delta))),
            float(np.max(np.abs(
                transformed_X[:, 2:4] - X[:, 2:4] @ matrix.T
            ))),
            float(np.max(np.abs(
                transformed_y[:, 2:4] - y[:, 2:4] @ matrix.T
            ))),
        )
        inverse_id = next(
            index for index, candidate in enumerate(D4_MATRICES)
            if np.array_equal(candidate, matrix.T)
        )
        inverse_ids = np.full(len(X), inverse_id, dtype=np.int64)
        round_X, round_y = augment_d4_batch(
            transformed_X, transformed_y, rng, transform_ids=inverse_ids
        )
        max_round_trip_error = max(
            max_round_trip_error,
            float(np.max(np.abs(round_X - X))),
            float(np.max(np.abs(round_y - y))),
        )
        if obstacle_count:
            original_obstacles = X[:, 6:].reshape(len(X), obstacle_count, 3)
            transformed_obstacles = transformed_X[:, 6:].reshape(
                len(X), obstacle_count, 3
            )
            padding = original_obstacles[:, :, 2] <= 0.0
            padding_preserved = padding_preserved and bool(np.all(
                transformed_obstacles[padding] == 0.0
            ))
            if not np.array_equal(
                transformed_obstacles[:, :, 2], original_obstacles[:, :, 2]
            ):
                raise RuntimeError("D4 augmentation changed obstacle radii")

    if not padding_preserved:
        raise RuntimeError("D4 augmentation changed zero-padding obstacle triples")
    worst_error = max(
        max_distance_error,
        max_speed_error,
        max_vector_error,
        max_round_trip_error,
    )
    if worst_error > 3e-6:
        raise RuntimeError(
            "D4 augmentation failed its numerical invariance audit: "
            f"worst_error={worst_error:.9g}"
        )
    return {
        "rows_checked": int(len(X)),
        "symmetries_checked": int(len(D4_MATRICES)),
        "maximum_orthogonality_error": orthogonality_error,
        "maximum_sample_distance_error": max_distance_error,
        "maximum_speed_error": max_speed_error,
        "maximum_transformed_vector_error": max_vector_error,
        "maximum_round_trip_error": max_round_trip_error,
        "padding_obstacles_preserved": bool(padding_preserved),
        "validation_and_testing_augmented": False,
    }


def params_to_state_np(X, params, bounds=None):
    del bounds  # Physical limits are enforced by the RRT steering/safety layer.
    dpos = params[:, 0:2].astype(np.float32)
    vel = params[:, 2:4].astype(np.float32)
    pos = X[:, 0:2].astype(np.float32) + dpos
    return np.concatenate([pos, vel], axis=1).astype(np.float32)


def params_to_state_torch(raw_x, p_norm, tnorm):
    p = p_norm * tnorm["p_std"] + tnorm["p_mean"]
    dpos = p[:, 0:2]
    pos = raw_x[:, 0:2] + dpos
    vel = p[:, 2:4]
    state = torch.cat([pos, vel], dim=1)
    return state, p


def canonical_params_norm_torch(p_norm, tnorm):
    """Return the lossless physical local-motion parameterization used by V7."""
    p = p_norm * tnorm["p_std"] + tnorm["p_mean"]
    return p_norm, p


def clip_vector_norm_np(vectors, max_norm=1.0):
    """Clip vector magnitudes without changing direction."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scales = np.minimum(
        1.0,
        float(max_norm) / np.maximum(norms, 1e-12),
    )
    return (vectors * scales).astype(np.float32)


def dynamics_velocity_from_delta_np(
    parent_velocity, delta_position, motion_dt, max_speed=1.0,
):
    """Recover next velocity from exact one-step double-integrator dynamics.

    The expert uses ``p1-p0 = 0.5 * (v0+v1_preclip) * dt`` and then clips
    only the saved next-velocity magnitude.  Therefore displacement and the
    parent velocity determine the expert next velocity up to float32 error.
    """
    motion_dt = float(motion_dt)
    if not math.isfinite(motion_dt) or motion_dt <= 0.0:
        raise ValueError("motion_dt must be positive and finite")
    parent_velocity = np.asarray(parent_velocity, dtype=np.float32)
    delta_position = np.asarray(delta_position, dtype=np.float32)
    preclip = 2.0 * delta_position / motion_dt - parent_velocity
    return clip_vector_norm_np(preclip, max_norm=max_speed)


def dynamics_velocity_from_delta_torch(
    parent_velocity, delta_position, motion_dt, max_speed,
):
    """Differentiable torch equivalent of dynamics_velocity_from_delta_np."""
    preclip = 2.0 * delta_position / motion_dt - parent_velocity
    speed = torch.norm(preclip, dim=1, keepdim=True)
    scale = torch.clamp(max_speed / (speed + 1e-12), max=1.0)
    return preclip * scale


def control_increment_from_delta_np(parent_velocity, delta_position, motion_dt):
    """Recover the pre-speed-clip normalized velocity increment.

    For one constant-acceleration step,

        delta_position = (parent_velocity + 0.5 * delta_velocity) * dt.

    Predicting ``delta_velocity`` directly avoids multiplying a learned
    displacement error by ``2 / dt`` when the next velocity is reconstructed.
    """
    motion_dt = float(motion_dt)
    if not math.isfinite(motion_dt) or motion_dt <= 0.0:
        raise ValueError("motion_dt must be positive and finite")
    parent_velocity = np.asarray(parent_velocity, dtype=np.float32)
    delta_position = np.asarray(delta_position, dtype=np.float32)
    return (
        2.0 * delta_position / motion_dt - 2.0 * parent_velocity
    ).astype(np.float32)


def params_from_control_increment_np(
    parent_velocity, control_increment, motion_dt, max_speed=1.0,
):
    """Map a normalized one-step velocity increment to [delta,next_velocity]."""
    motion_dt = float(motion_dt)
    parent_velocity = np.asarray(parent_velocity, dtype=np.float32)
    control_increment = np.asarray(control_increment, dtype=np.float32)
    delta_position = (
        parent_velocity + 0.5 * control_increment
    ) * motion_dt
    next_velocity_preclip = parent_velocity + control_increment
    next_velocity = clip_vector_norm_np(
        next_velocity_preclip, max_norm=max_speed
    )
    return np.concatenate(
        [delta_position, next_velocity], axis=1
    ).astype(np.float32)


def params_from_control_increment_torch(
    parent_velocity, control_increment, motion_dt, max_speed,
):
    """Differentiable torch equivalent of params_from_control_increment_np."""
    delta_position = (
        parent_velocity + 0.5 * control_increment
    ) * motion_dt
    next_velocity_preclip = parent_velocity + control_increment
    speed = torch.norm(next_velocity_preclip, dim=1, keepdim=True)
    scale = torch.clamp(max_speed / (speed + 1e-12), max=1.0)
    next_velocity = next_velocity_preclip * scale
    return torch.cat([delta_position, next_velocity], dim=1)


def infer_effective_motion_dt(train_X, train_params):
    """Infer normalized dt from unsaturated labels as a defensive fallback."""
    delta = np.asarray(train_params[:, 0:2], dtype=np.float64)
    parent_velocity = np.asarray(train_X[:, 2:4], dtype=np.float64)
    next_velocity = np.asarray(train_params[:, 2:4], dtype=np.float64)
    velocity_sum = parent_velocity + next_velocity
    denominator = np.sum(velocity_sum * velocity_sum, axis=1)
    unsaturated = np.linalg.norm(next_velocity, axis=1) < 0.999
    valid = unsaturated & (denominator > 1e-12)
    if not np.any(valid):
        raise RuntimeError(
            "Cannot infer the normalized one-step interval from training labels"
        )
    estimates = (
        2.0 * np.sum(delta[valid] * velocity_sum[valid], axis=1)
        / denominator[valid]
    )
    estimates = estimates[np.isfinite(estimates) & (estimates > 0.0)]
    if len(estimates) == 0:
        raise RuntimeError(
            "No positive finite normalized one-step interval could be inferred"
        )
    return float(np.median(estimates))


def fit_normalizer(train_X, train_params, motion_dt=None):
    x_mean = train_X.mean(axis=0).astype(np.float32)
    x_std = train_X.std(axis=0).astype(np.float32)
    local_params = motion_params_in_sample_frame(train_X, train_params)
    # Balance all vector normalizers under the same D4 group used for training.
    # This removes orientation-dependent scaling without reading held-out data.
    p_mean = np.zeros(4, dtype=np.float32)
    p_std = np.asarray(
        [
            np.sqrt(np.mean(train_params[:, 0:2] ** 2)),
            np.sqrt(np.mean(train_params[:, 0:2] ** 2)),
            np.sqrt(np.mean(train_params[:, 2:4] ** 2)),
            np.sqrt(np.mean(train_params[:, 2:4] ** 2)),
        ],
        dtype=np.float32,
    )
    local_p_mean = np.asarray(
        [
            np.mean(local_params[:, 0]), 0.0,
            np.mean(local_params[:, 2]), 0.0,
        ],
        dtype=np.float32,
    )
    local_p_std = np.sqrt(
        np.mean((local_params - local_p_mean[None, :]) ** 2, axis=0)
    ).astype(np.float32)
    x_std = np.where(np.abs(x_std) < 1e-8, 1.0, x_std)
    p_std = np.where(np.abs(p_std) < 1e-8, 1.0, p_std)
    local_p_std = np.where(np.abs(local_p_std) < 1e-8, 1.0, local_p_std)
    step = np.linalg.norm(train_params[:, 0:2], axis=1)
    speed = np.linalg.norm(train_params[:, 2:4], axis=1)
    step_min = float(max(0.0, np.percentile(step, 0.1) * 0.70))
    step_max = float(max(step_min + 1e-5, np.percentile(step, 99.9) * 1.30))
    speed_min = float(max(0.0, np.percentile(speed, 0.1) * 0.70))
    speed_max = float(max(speed_min + 1e-5, np.percentile(speed, 99.9) * 1.30))
    if motion_dt is None:
        motion_dt = infer_effective_motion_dt(train_X, train_params)
    median_label_dt = float(motion_dt)
    if not math.isfinite(median_label_dt) or median_label_dt <= 0.0:
        raise ValueError("The normalized one-step interval must be positive and finite")
    control_increment = control_increment_from_delta_np(
        train_X[:, 2:4], train_params[:, 0:2], median_label_dt
    )
    local_control_increment = motion_params_in_sample_frame(
        train_X,
        np.concatenate(
            [np.zeros_like(control_increment), control_increment], axis=1
        ),
    )[:, 2:4]
    control_scale = float(np.sqrt(np.mean(control_increment ** 2)))
    control_increment_mean = np.zeros(2, dtype=np.float32)
    control_increment_std = np.full(2, control_scale, dtype=np.float32)
    local_control_increment_mean = np.asarray(
        [np.mean(local_control_increment[:, 0]), 0.0], dtype=np.float32
    )
    local_control_increment_std = np.sqrt(
        np.mean(
            (
                local_control_increment
                - local_control_increment_mean[None, :]
            ) ** 2,
            axis=0,
        )
    ).astype(np.float32)
    control_increment_std = np.where(
        np.abs(control_increment_std) < 1e-8, 1.0, control_increment_std
    ).astype(np.float32)
    local_control_increment_std = np.where(
        np.abs(local_control_increment_std) < 1e-8,
        1.0,
        local_control_increment_std,
    ).astype(np.float32)
    # This is learned from the training split only. The small guard accommodates
    # float32 roundoff while retaining the expert componentwise acceleration
    # envelope on held-out rows generated under the same contract.
    control_increment_component_limit = float(
        max(1e-4, np.max(np.abs(control_increment)) * 1.001)
    )
    dynamics_prior_velocity = dynamics_velocity_from_delta_np(
        train_X[:, 2:4], train_params[:, 0:2], median_label_dt,
        max_speed=1.0,
    )
    kinematic_error = np.linalg.norm(
        train_params[:, 2:4] - dynamics_prior_velocity,
        axis=1,
    )
    kinematic_error_p95 = float(np.percentile(kinematic_error, 95))
    speed_scale = float(max(np.percentile(speed, 95), 0.05))
    kinematic_prior_strength = float(np.clip(
        1.0 - 2.0 * kinematic_error_p95 / speed_scale, 0.0, 1.0
    ))
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "p_mean": p_mean,
        "p_std": p_std,
        "local_p_mean": local_p_mean,
        "local_p_std": local_p_std,
        "control_increment_mean": control_increment_mean,
        "control_increment_std": control_increment_std,
        "local_control_increment_mean": local_control_increment_mean,
        "local_control_increment_std": local_control_increment_std,
        "control_increment_component_limit": (
            control_increment_component_limit
        ),
        "step_min": step_min,
        "step_max": step_max,
        "speed_min": speed_min,
        "speed_max": speed_max,
        "median_label_dt": median_label_dt,
        "normalized_speed_limit": 1.0,
        "kinematic_prior_strength": kinematic_prior_strength,
        "kinematic_prior_p95_velocity_error": kinematic_error_p95,
        "kinematic_prior_mean_velocity_error": float(np.mean(kinematic_error)),
    }


def norm_X(X, norm):
    return ((X.astype(np.float32) - norm["x_mean"]) / norm["x_std"]).astype(np.float32)


def norm_params(P, norm):
    return ((P.astype(np.float32) - norm["p_mean"]) / norm["p_std"]).astype(np.float32)


def unnorm_params(Pn, norm):
    return (Pn.astype(np.float32) * norm["p_std"] + norm["p_mean"]).astype(np.float32)


def tensor_normalizer(norm, device):
    return {
        "p_mean": torch.tensor(norm["p_mean"], dtype=torch.float32, device=device),
        "p_std": torch.tensor(norm["p_std"], dtype=torch.float32, device=device),
        "control_increment_mean": torch.tensor(
            norm["control_increment_mean"], dtype=torch.float32, device=device
        ),
        "control_increment_std": torch.tensor(
            norm["control_increment_std"], dtype=torch.float32, device=device
        ),
        "step_min": torch.tensor(norm["step_min"], dtype=torch.float32, device=device),
        "step_max": torch.tensor(norm["step_max"], dtype=torch.float32, device=device),
        "speed_min": torch.tensor(norm["speed_min"], dtype=torch.float32, device=device),
        "speed_max": torch.tensor(norm["speed_max"], dtype=torch.float32, device=device),
        "median_label_dt": torch.tensor(norm["median_label_dt"], dtype=torch.float32, device=device),
        "normalized_speed_limit": torch.tensor(
            norm["normalized_speed_limit"], dtype=torch.float32, device=device
        ),
        "kinematic_prior_strength": torch.tensor(
            norm["kinematic_prior_strength"], dtype=torch.float32, device=device
        ),
    }


def estimate_dt_from_labels(X, y):
    dpos = y[:, 0:2] - X[:, 0:2]
    step = np.linalg.norm(dpos, axis=1)
    speed = np.linalg.norm(y[:, 2:4], axis=1)
    mask = (step > 1e-8) & (speed > 1e-8)
    if not np.any(mask):
        return None
    return float(np.median(step[mask] / speed[mask]))


def cosine_np(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-8)


def dataset_diagnostics(data, include_testing=True):
    rows = []
    effective_motion_dt = float(
        data["supervision_lookahead_steps"]
        * data["nominal_dt"]
        * data["velocity_scale"]
        / float(data["position_scale"][0])
    )
    split_specs = [
        ("training", "train_X", "train_y"),
        ("validation", "validation_X", "validation_y"),
    ]
    if include_testing:
        split_specs.append(("testing", "testing_X", "testing_y"))
    for split, xkey, ykey in split_specs:
        X = data[xkey]
        y = data[ykey]
        dpos = y[:, 0:2] - X[:, 0:2]
        step = np.linalg.norm(dpos, axis=1)
        speed = np.linalg.norm(y[:, 2:4], axis=1)
        valid_angle = (step > 1e-8) & (speed > 1e-8)
        cos = cosine_np(dpos, y[:, 2:4])
        angle = np.degrees(np.arccos(np.clip(cos[valid_angle], -1.0, 1.0)))
        dt_est = estimate_dt_from_labels(X, y)
        if dt_est is not None and dt_est > 1e-8:
            kinematic_velocity_error = np.linalg.norm(
                y[:, 2:4] - dpos / dt_est, axis=1
            )
        else:
            kinematic_velocity_error = np.asarray([], dtype=np.float32)
        exact_dynamics_velocity = dynamics_velocity_from_delta_np(
            X[:, 2:4], dpos, effective_motion_dt, max_speed=1.0
        )
        exact_dynamics_velocity_error = np.linalg.norm(
            y[:, 2:4] - exact_dynamics_velocity, axis=1
        )
        control_increment = control_increment_from_delta_np(
            X[:, 2:4], dpos, effective_motion_dt
        )
        rows.append({
            "split": split,
            "rows": int(len(X)),
            "mean_step_length": float(np.mean(step)),
            "p90_step_length": float(np.percentile(step, 90)),
            "p95_step_length": float(np.percentile(step, 95)),
            "p99_step_length": float(np.percentile(step, 99)),
            "mean_next_speed": float(np.mean(speed)),
            "p90_next_speed": float(np.percentile(speed, 90)),
            "p95_next_speed": float(np.percentile(speed, 95)),
            "p99_next_speed": float(np.percentile(speed, 99)),
            "median_dt_from_labels": float(dt_est) if dt_est is not None else None,
            "mean_delta_over_dt_velocity_error": float(
                np.mean(kinematic_velocity_error)
            ) if len(kinematic_velocity_error) else None,
            "p95_delta_over_dt_velocity_error": float(
                np.percentile(kinematic_velocity_error, 95)
            ) if len(kinematic_velocity_error) else None,
            "exact_double_integrator_effective_dt": effective_motion_dt,
            "mean_exact_dynamics_velocity_reconstruction_error": float(
                np.mean(exact_dynamics_velocity_error)
            ),
            "p95_exact_dynamics_velocity_reconstruction_error": float(
                np.percentile(exact_dynamics_velocity_error, 95)
            ),
            "max_exact_dynamics_velocity_reconstruction_error": float(
                np.max(exact_dynamics_velocity_error)
            ),
            "mean_control_increment_norm": float(
                np.mean(np.linalg.norm(control_increment, axis=1))
            ),
            "p95_control_increment_norm": float(
                np.percentile(np.linalg.norm(control_increment, axis=1), 95)
            ),
            "max_abs_control_increment_component": float(
                np.max(np.abs(control_increment))
            ),
            "rows_with_defined_step_velocity_angle": int(np.sum(valid_angle)),
            "mean_step_velocity_angle_deg": float(np.mean(angle)) if len(angle) else None,
            "p90_step_velocity_angle_deg": float(np.percentile(angle, 90)) if len(angle) else None,
            "pct_angle_gt_45deg": float(100.0 * np.mean(angle > 45.0)) if len(angle) else None,
            "pct_angle_gt_90deg": float(100.0 * np.mean(angle > 90.0)) if len(angle) else None,
        })
    return rows


def representation_ceiling(
    data, norm, state_threshold=0.10, position_threshold=0.05,
    velocity_threshold=0.05, include_testing=True,
):
    out = {}
    split_specs = [
        ("training", "train_X", "train_y"),
        ("validation", "validation_X", "validation_y"),
    ]
    if include_testing:
        split_specs.append(("testing", "testing_X", "testing_y"))
    for split, xkey, ykey in split_specs:
        X = data[xkey]
        y = data[ykey]
        P = target_direction_step_speed(X, y)
        # Do not clip here: this measures only information lost by the
        # direction/step/speed representation, not chosen training bounds.
        pred = params_to_state_np(X, P, bounds=None)
        err = np.linalg.norm(pred - y[:, 0:4], axis=1)
        pos_err = np.linalg.norm(pred[:, 0:2] - y[:, 0:2], axis=1)
        vel_err = np.linalg.norm(pred[:, 2:4] - y[:, 2:4], axis=1)
        out[split] = {
            "ceiling_full_state_accuracy_0_10_percent": float(100.0 * np.mean(err <= state_threshold)),
            "ceiling_position_accuracy_0_05_percent": float(100.0 * np.mean(pos_err <= position_threshold)),
            "ceiling_velocity_accuracy_0_05_percent": float(100.0 * np.mean(vel_err <= velocity_threshold)),
            "full_state_threshold_used": float(state_threshold),
            "position_threshold_used": float(position_threshold),
            "velocity_threshold_used": float(velocity_threshold),
            "ceiling_mean_full_state_error": float(np.mean(err)),
            "ceiling_p95_full_state_error": float(np.percentile(err, 95)),
        }
    return out


def condition_ambiguity_diagnostics(
    data, norm, decimals=4, conflict_threshold=0.10, include_testing=True,
):
    """Find near-identical conditions that carry meaningfully different labels.

    This is a dataset ambiguity diagnostic, not a claimed neural-network upper
    bound. Conditions are normalized and rounded only for grouping.
    """
    out = {}
    split_specs = [
        ("training", "train_X", "train_y"),
        ("validation", "validation_X", "validation_y"),
    ]
    if include_testing:
        split_specs.append(("testing", "testing_X", "testing_y"))
    for split, xkey, ykey in split_specs:
        Xq = np.round(norm_X(data[xkey], norm), decimals=decimals)
        y = data[ykey][:, 0:4]
        _, inverse, counts = np.unique(Xq, axis=0, return_inverse=True, return_counts=True)
        repeated_ids = np.flatnonzero(counts > 1)
        conflicting_groups = 0
        conflicting_rows = 0
        worst_range = 0.0
        for group_id in repeated_ids:
            idx = np.flatnonzero(inverse == group_id)
            labels = y[idx]
            range_diagonal = float(np.linalg.norm(labels.max(axis=0) - labels.min(axis=0)))
            worst_range = max(worst_range, range_diagonal)
            if range_diagonal > conflict_threshold:
                conflicting_groups += 1
                conflicting_rows += len(idx)
        repeated_rows = int(np.sum(counts[counts > 1]))
        out[split] = {
            "rounding_decimals_in_normalized_condition_space": int(decimals),
            "rows": int(len(Xq)),
            "repeated_condition_groups": int(len(repeated_ids)),
            "rows_in_repeated_condition_groups": repeated_rows,
            "conflicting_condition_groups": int(conflicting_groups),
            "rows_in_conflicting_condition_groups": int(conflicting_rows),
            "conflicting_row_percent": float(100.0 * conflicting_rows / max(1, len(Xq))),
            "worst_group_target_range_diagonal": float(worst_range),
            "conflict_threshold": float(conflict_threshold),
        }
    return out


class LocalMotionGenerator(nn.Module):
    """D4-equivariant, permutation-invariant, physics-informed generator.

    Learned layers never receive raw absolute parent/sample coordinates or a
    flattened obstacle map. They receive sample-frame motion features and a set
    of local inequality-constraint tokens. Consequently, translating an
    obstacle environment cannot act as an environment identifier.
    """

    constraint_token_dim = 18
    constraint_embedding_dim = 96
    attention_heads = 4

    def __init__(self, cond_dim, norm, noise_dim=16):
        super().__init__()
        if cond_dim < 6 or (cond_dim - 6) % 3 != 0:
            raise ValueError("Generator received an invalid condition dimension")
        if self.constraint_embedding_dim % self.attention_heads != 0:
            raise RuntimeError("Constraint embedding must divide attention heads")
        self.cond_dim = int(cond_dim)
        self.noise_dim = int(noise_dim)
        self.obstacles_used_by_generator = (cond_dim - 6) // 3
        self.uses_raw_absolute_coordinate_features = False
        self.uses_permutation_invariant_constraint_set = True

        self.register_buffer(
            "condition_mean", torch.as_tensor(norm["x_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "condition_std", torch.as_tensor(norm["x_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "motion_p_mean", torch.as_tensor(norm["p_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "motion_p_std", torch.as_tensor(norm["p_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "local_p_mean", torch.as_tensor(norm["local_p_mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "local_p_std", torch.as_tensor(norm["local_p_std"], dtype=torch.float32)
        )
        self.register_buffer(
            "local_control_increment_mean",
            torch.as_tensor(
                norm["local_control_increment_mean"], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "local_control_increment_std",
            torch.as_tensor(
                norm["local_control_increment_std"], dtype=torch.float32
            ),
        )
        self.register_buffer(
            "control_increment_component_limit",
            torch.as_tensor(
                float(norm["control_increment_component_limit"]),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "motion_dt",
            torch.as_tensor(
                max(float(norm["median_label_dt"]), 1e-6),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "normalized_speed_limit",
            torch.as_tensor(
                float(norm["normalized_speed_limit"]), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "kinematic_prior_strength",
            torch.as_tensor(
                float(norm["kinematic_prior_strength"]), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "physical_acceleration_to_increment",
            torch.as_tensor(
                PHYSICAL_TIME_STEP / PHYSICAL_VELOCITY_SCALE,
                dtype=torch.float32,
            ),
        )

        context_dim = 10 + self.noise_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.constraint_encoder = nn.Sequential(
            nn.Linear(self.constraint_token_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.constraint_embedding_dim),
            nn.LayerNorm(self.constraint_embedding_dim),
            nn.SiLU(),
        )
        self.attention_query = nn.Linear(128, self.constraint_embedding_dim)
        self.constraint_self_query = nn.Linear(
            self.constraint_embedding_dim, self.constraint_embedding_dim
        )
        self.constraint_self_key = nn.Linear(
            self.constraint_embedding_dim, self.constraint_embedding_dim
        )
        self.constraint_self_value = nn.Linear(
            self.constraint_embedding_dim, self.constraint_embedding_dim
        )
        self.constraint_self_output = nn.Linear(
            self.constraint_embedding_dim, self.constraint_embedding_dim
        )
        self.constraint_interaction_norm = nn.LayerNorm(
            self.constraint_embedding_dim
        )
        self.attention_key = nn.Linear(
            self.constraint_embedding_dim, self.constraint_embedding_dim
        )
        self.attention_value = nn.Linear(
            self.constraint_embedding_dim, self.constraint_embedding_dim
        )
        pooled_dim = (
            128
            + self.constraint_embedding_dim
            + self.constraint_embedding_dim
            + self.constraint_embedding_dim
        )
        self.control_increment_residual = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )
        # The untrained network is the verified label-free nominal LQR law.
        # Training learns only obstacle/boundary corrections to that prior.
        nn.init.zeros_(self.control_increment_residual[-1].weight)
        nn.init.zeros_(self.control_increment_residual[-1].bias)

    @staticmethod
    def _to_local(vector, sample_unit, sample_perpendicular):
        return torch.stack(
            [
                torch.sum(vector * sample_unit, dim=-1),
                torch.sum(vector * sample_perpendicular, dim=-1),
            ],
            dim=-1,
        )

    def _constraint_features(
        self,
        normals,
        boundaries,
        clearances,
        radii,
        distances,
        type_index,
        active,
        sample_unit,
        sample_perpendicular,
        objective_control_increment,
        baseline_control_increment,
    ):
        normal_length = torch.norm(normals, dim=2)
        valid = active & (normal_length > 1e-8)
        safe_length = torch.clamp(normal_length, min=1e-8)
        normal_unit = normals / safe_length[:, :, None]
        normal_local = torch.stack(
            [
                torch.sum(normal_unit * sample_unit[:, None, :], dim=2),
                torch.sum(
                    normal_unit * sample_perpendicular[:, None, :], dim=2
                ),
            ],
            dim=2,
        )
        rhs_increment = (
            boundaries / safe_length * self.physical_acceleration_to_increment
        )
        objective_projection = torch.sum(
            normal_unit * objective_control_increment[:, None, :], dim=2
        )
        objective_slack = objective_projection - rhs_increment
        baseline_projection = torch.sum(
            normal_unit * baseline_control_increment[:, None, :], dim=2
        )
        baseline_slack = baseline_projection - rhs_increment
        scale = torch.clamp(self.control_increment_component_limit, min=1e-6)
        single_constraint_correction_local = (
            torch.relu(-objective_slack)[:, :, None] * normal_local / scale
        )
        type_features = torch.zeros(
            len(normals), len(normals[0]), 3,
            dtype=normals.dtype,
            device=normals.device,
        )
        type_features[:, :, int(type_index)] = 1.0
        features = torch.cat(
            [
                normal_local,
                torch.clamp(rhs_increment[:, :, None] / scale, -8.0, 8.0),
                torch.clamp(
                    objective_projection[:, :, None] / scale, -12.0, 12.0
                ),
                torch.clamp(
                    objective_slack[:, :, None] / scale, -12.0, 12.0
                ),
                torch.clamp(
                    torch.relu(-objective_slack)[:, :, None] / scale,
                    0.0,
                    12.0,
                ),
                torch.clamp(
                    single_constraint_correction_local, -12.0, 12.0
                ),
                torch.clamp(
                    baseline_projection[:, :, None] / scale, -8.0, 8.0
                ),
                torch.clamp(
                    baseline_slack[:, :, None] / scale, -8.0, 8.0
                ),
                torch.clamp(
                    torch.relu(-baseline_slack)[:, :, None] / scale,
                    0.0,
                    8.0,
                ),
                torch.tanh(torch.log1p(normal_length)[:, :, None] / 4.0),
                torch.clamp(clearances[:, :, None] / 0.20, -4.0, 6.0),
                torch.clamp(radii[:, :, None] / 0.20, 0.0, 6.0),
                torch.clamp(distances[:, :, None], 0.0, 3.0),
                type_features,
            ],
            dim=2,
        )
        return features * valid[:, :, None].to(features.dtype), valid

    def _build_constraint_tokens(
        self,
        raw_cond,
        sample_unit,
        sample_perpendicular,
        objective_control_increment,
        baseline_control_increment,
    ):
        batch = len(raw_cond)
        dtype = raw_cond.dtype
        device = raw_cond.device
        position = raw_cond[:, 0:2] * PHYSICAL_POSITION_SCALE
        velocity = raw_cond[:, 2:4] * PHYSICAL_VELOCITY_SCALE
        token_parts = []
        mask_parts = []

        if self.obstacles_used_by_generator:
            obstacles = raw_cond[:, 6:].reshape(
                batch, self.obstacles_used_by_generator, 3
            )
            centers = obstacles[:, :, 0:2] * PHYSICAL_POSITION_SCALE
            obstacle_radii = obstacles[:, :, 2] * PHYSICAL_POSITION_SCALE
            relative = position[:, None, :] - centers
            center_distance = torch.norm(relative, dim=2)
            combined_radius = obstacle_radii + PHYSICAL_CONFIGURATION_RADIUS
            barrier = center_distance.square() - combined_radius.square()
            barrier_dot = 2.0 * torch.sum(
                relative * velocity[:, None, :], dim=2
            )
            obstacle_normals = 2.0 * relative
            obstacle_boundaries = (
                -2.0 * torch.sum(velocity.square(), dim=1)[:, None]
                - CBF_K1 * barrier_dot
                - CBF_K0 * barrier
            )
            obstacle_clearance = (
                center_distance - combined_radius
            ) / PHYSICAL_POSITION_SCALE
            obstacle_tokens, obstacle_mask = self._constraint_features(
                obstacle_normals,
                obstacle_boundaries,
                obstacle_clearance,
                obstacle_radii / PHYSICAL_POSITION_SCALE,
                center_distance / PHYSICAL_POSITION_SCALE,
                0,
                obstacle_radii > 0.0,
                sample_unit,
                sample_perpendicular,
                objective_control_increment,
                baseline_control_increment,
            )
            token_parts.append(obstacle_tokens)
            mask_parts.append(obstacle_mask)

        base_normals = raw_cond.new_tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
        boundary_normals = base_normals[None, :, :].expand(batch, -1, -1)
        px, py = position[:, 0], position[:, 1]
        vx, vy = velocity[:, 0], velocity[:, 1]
        width = PHYSICAL_POSITION_SCALE
        boundary_rhs = torch.stack(
            [
                -CBF_K1 * vx - CBF_K0 * (px - PHYSICAL_CONFIGURATION_RADIUS),
                CBF_K1 * vx
                - CBF_K0 * (width - PHYSICAL_CONFIGURATION_RADIUS - px),
                -CBF_K1 * vy - CBF_K0 * (py - PHYSICAL_CONFIGURATION_RADIUS),
                CBF_K1 * vy
                - CBF_K0 * (width - PHYSICAL_CONFIGURATION_RADIUS - py),
            ],
            dim=1,
        )
        boundary_clearance = torch.stack(
            [
                px - PHYSICAL_CONFIGURATION_RADIUS,
                width - PHYSICAL_CONFIGURATION_RADIUS - px,
                py - PHYSICAL_CONFIGURATION_RADIUS,
                width - PHYSICAL_CONFIGURATION_RADIUS - py,
            ],
            dim=1,
        ) / PHYSICAL_POSITION_SCALE
        always_active = torch.ones(
            batch, 4, dtype=torch.bool, device=device
        )
        zeros = torch.zeros(batch, 4, dtype=dtype, device=device)
        boundary_tokens, boundary_mask = self._constraint_features(
            boundary_normals,
            boundary_rhs,
            boundary_clearance,
            zeros,
            torch.clamp(boundary_clearance, min=0.0),
            1,
            always_active,
            sample_unit,
            sample_perpendicular,
            objective_control_increment,
            baseline_control_increment,
        )
        token_parts.append(boundary_tokens)
        mask_parts.append(boundary_mask)

        box_rhs = torch.full(
            (batch, 4),
            -PHYSICAL_MAX_ACCELERATION,
            dtype=dtype,
            device=device,
        )
        box_tokens, box_mask = self._constraint_features(
            boundary_normals,
            box_rhs,
            zeros,
            zeros,
            zeros,
            2,
            always_active,
            sample_unit,
            sample_perpendicular,
            objective_control_increment,
            baseline_control_increment,
        )
        token_parts.append(box_tokens)
        mask_parts.append(box_mask)
        return torch.cat(token_parts, dim=1), torch.cat(mask_parts, dim=1)

    def _predict_local_correction(
        self, context_raw, constraint_tokens, constraint_mask
    ):
        context = self.context_encoder(context_raw)
        embeddings = self.constraint_encoder(constraint_tokens)
        embeddings = embeddings * constraint_mask[:, :, None].to(
            embeddings.dtype
        )
        batch, token_count, embedding_dim = embeddings.shape
        head_dim = embedding_dim // self.attention_heads
        self_queries = self.constraint_self_query(embeddings).reshape(
            batch, token_count, self.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        self_keys = self.constraint_self_key(embeddings).reshape(
            batch, token_count, self.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        self_values = self.constraint_self_value(embeddings).reshape(
            batch, token_count, self.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        interaction_scores = torch.matmul(
            self_queries, self_keys.transpose(2, 3)
        ) / math.sqrt(float(head_dim))
        interaction_scores = interaction_scores.masked_fill(
            ~constraint_mask[:, None, None, :],
            torch.finfo(interaction_scores.dtype).min,
        )
        interaction_weights = torch.softmax(interaction_scores, dim=3)
        interactions = torch.matmul(
            interaction_weights, self_values
        ).permute(0, 2, 1, 3).reshape(batch, token_count, embedding_dim)
        embeddings = self.constraint_interaction_norm(
            embeddings + self.constraint_self_output(interactions)
        )
        embeddings = embeddings * constraint_mask[:, :, None].to(
            embeddings.dtype
        )
        query = self.attention_query(context).reshape(
            batch, self.attention_heads, head_dim
        )
        keys = self.attention_key(embeddings).reshape(
            batch, token_count, self.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        values = self.attention_value(embeddings).reshape(
            batch, token_count, self.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        scores = torch.sum(query[:, :, None, :] * keys, dim=3) / math.sqrt(
            float(head_dim)
        )
        scores = scores.masked_fill(
            ~constraint_mask[:, None, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores, dim=2)
        attended = torch.sum(attention[:, :, :, None] * values, dim=2).reshape(
            batch, embedding_dim
        )
        mask_float = constraint_mask[:, :, None].to(embeddings.dtype)
        mean_pooled = torch.sum(embeddings, dim=1) / torch.clamp(
            torch.sum(mask_float, dim=1), min=1.0
        )
        max_pooled = embeddings.masked_fill(
            ~constraint_mask[:, :, None], torch.finfo(embeddings.dtype).min
        ).max(dim=1).values
        return self.control_increment_residual(
            torch.cat([context, attended, mean_pooled, max_pooled], dim=1)
        )

    def forward(self, cond, noise):
        if cond.ndim != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError("Generator condition tensor has the wrong shape")
        if noise.ndim != 2 or noise.shape != (len(cond), self.noise_dim):
            raise ValueError("Generator noise tensor has the wrong shape")
        raw_cond = cond * self.condition_std + self.condition_mean
        to_sample = raw_cond[:, 4:6] - raw_cond[:, 0:2]
        sample_distance = torch.norm(to_sample, dim=1, keepdim=True)
        sample_unit = to_sample / torch.clamp(sample_distance, min=1e-8)
        default_unit = torch.zeros_like(sample_unit)
        default_unit[:, 0] = 1.0
        sample_unit = torch.where(
            sample_distance > 1e-8, sample_unit, default_unit
        )
        sample_perpendicular = torch.stack(
            [-sample_unit[:, 1], sample_unit[:, 0]], dim=1
        )
        parent_velocity = raw_cond[:, 2:4]
        parent_velocity_local = self._to_local(
            parent_velocity, sample_unit, sample_perpendicular
        )
        parent_speed = torch.norm(parent_velocity, dim=1, keepdim=True)
        clipped_target_distance = torch.clamp(
            sample_distance, max=NOMINAL_LQR_STEP_DISTANCE_NORMALIZED
        )
        nominal_target_delta = sample_unit * clipped_target_distance
        nominal_unclipped_global = (
            NOMINAL_LQR_POSITION_INCREMENT_GAIN * nominal_target_delta
            - NOMINAL_LQR_VELOCITY_INCREMENT_GAIN * parent_velocity
        )
        nominal_clipped_global = torch.clamp(
            nominal_unclipped_global,
            min=-self.control_increment_component_limit,
            max=self.control_increment_component_limit,
        )
        nominal_unclipped_local = self._to_local(
            nominal_unclipped_global, sample_unit, sample_perpendicular
        )
        nominal_clipped_local = self._to_local(
            nominal_clipped_global, sample_unit, sample_perpendicular
        )
        control_scale = torch.clamp(
            self.control_increment_component_limit, min=1e-6
        )
        distance_scale = NOMINAL_LQR_STEP_DISTANCE_NORMALIZED
        context_raw = torch.cat(
            [
                torch.clamp(sample_distance / distance_scale, 0.0, 8.0),
                clipped_target_distance / distance_scale,
                torch.clamp(
                    (sample_distance - clipped_target_distance) / distance_scale,
                    0.0,
                    8.0,
                ),
                parent_velocity_local,
                parent_speed,
                torch.clamp(
                    nominal_unclipped_local / control_scale, -12.0, 12.0
                ),
                torch.clamp(nominal_clipped_local / control_scale, -2.0, 2.0),
                noise,
            ],
            dim=1,
        )
        constraint_tokens, constraint_mask = self._build_constraint_tokens(
            raw_cond,
            sample_unit,
            sample_perpendicular,
            nominal_unclipped_global,
            nominal_clipped_global,
        )
        # In a reflected world, all sample-frame perpendicular components flip
        # sign while scalar constraint quantities remain unchanged. Evaluate
        # both parity views and average them after mapping the mirrored output
        # back. This enforces reflection equivariance for arbitrary weights;
        # the sample frame already enforces rotation equivariance.
        mirrored_context = context_raw.clone()
        mirrored_context[:, [4, 7, 9]] *= -1.0
        mirrored_tokens = constraint_tokens.clone()
        mirrored_tokens[:, :, [1, 7]] *= -1.0
        paired_context = torch.cat([context_raw, mirrored_context], dim=0)
        paired_tokens = torch.cat(
            [constraint_tokens, mirrored_tokens], dim=0
        )
        paired_mask = torch.cat([constraint_mask, constraint_mask], dim=0)
        paired_correction = self._predict_local_correction(
            paired_context, paired_tokens, paired_mask
        )
        original_correction, mirrored_correction = torch.chunk(
            paired_correction, 2, dim=0
        )
        mirrored_correction = mirrored_correction * raw_cond.new_tensor(
            [1.0, -1.0]
        )
        correction_norm = 0.5 * (
            original_correction + mirrored_correction
        )
        nominal_local_norm = (
            nominal_clipped_local - self.local_control_increment_mean
        ) / self.local_control_increment_std
        local_control_increment = (
            (nominal_local_norm + correction_norm)
            * self.local_control_increment_std
            + self.local_control_increment_mean
        )
        global_control_increment = (
            local_control_increment[:, 0:1] * sample_unit
            + local_control_increment[:, 1:2] * sample_perpendicular
        )
        global_control_increment = torch.clamp(
            global_control_increment,
            min=-self.control_increment_component_limit,
            max=self.control_increment_component_limit,
        )
        global_params = params_from_control_increment_torch(
            parent_velocity,
            global_control_increment,
            self.motion_dt,
            self.normalized_speed_limit,
        )
        return (global_params - self.motion_p_mean) / self.motion_p_std


def load_generator_state_for_version(model, state_dict, source_version):
    """Load only the identical V7.7 architecture; older shortcuts are banned."""
    if source_version != SCRIPT_VERSION:
        raise RuntimeError(
            "V7.7 cannot import an absolute-coordinate V7.5/V7.6 generator: "
            f"{source_version!r}"
        )
    model.load_state_dict(state_dict, strict=True)


class LocalMotionDiscriminator(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 4, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.LeakyReLU(0.2),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 1),
        )

    def forward(self, cond, params_norm):
        return self.net(torch.cat([cond, params_norm], dim=1)).view(-1)


def run_model_smoke_test(data, norm, args, device):
    """Exercise both networks and a backward pass before a long training run."""
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    )
    try:
        batch_size = min(8, len(data["train_X"]))
        if batch_size < 1:
            raise RuntimeError("Model smoke test requires a nonempty training split")
        generator = LocalMotionGenerator(
            data["train_X"].shape[1], norm, noise_dim=args.noise_dim
        ).to(device)
        discriminator = LocalMotionDiscriminator(
            data["train_X"].shape[1]
        ).to(device)
        xb = torch.tensor(
            norm_X(data["train_X"][:batch_size], norm),
            dtype=torch.float32,
            device=device,
        )
        raw_x = torch.tensor(
            data["train_X"][:batch_size], dtype=torch.float32, device=device
        )
        z = torch.zeros(batch_size, args.noise_dim, device=device)
        final_layer = generator.control_increment_residual[-1]
        if bool(torch.count_nonzero(final_layer.weight).item()) or bool(
            torch.count_nonzero(final_layer.bias).item()
        ):
            raise RuntimeError(
                "Generator residual does not start from the nominal LQR prior"
            )
        # Use arbitrary nonzero residual weights for structural invariance tests;
        # a zero residual would make those tests pass trivially.
        with torch.no_grad():
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(final_layer.bias, mean=0.0, std=0.01)
        generator.eval()
        with torch.no_grad():
            original_output = generator(xb, z)
            original_state, _ = params_to_state_torch(
                raw_x, original_output, tensor_normalizer(norm, device)
            )

            obstacle_count = (data["train_X"].shape[1] - 6) // 3
            if obstacle_count > 1:
                permuted_raw_np = data["train_X"][:batch_size].copy()
                obstacle_view = permuted_raw_np[:, 6:].reshape(
                    batch_size, obstacle_count, 3
                )
                permuted_raw_np[:, 6:] = obstacle_view[:, ::-1, :].reshape(
                    batch_size, -1
                )
                permuted_x = torch.as_tensor(
                    norm_X(permuted_raw_np, norm),
                    dtype=torch.float32,
                    device=device,
                )
                permuted_raw = torch.as_tensor(
                    permuted_raw_np, dtype=torch.float32, device=device
                )
                permuted_output = generator(permuted_x, z)
                permuted_state, _ = params_to_state_torch(
                    permuted_raw,
                    permuted_output,
                    tensor_normalizer(norm, device),
                )
                permutation_error = torch.max(
                    torch.abs(permuted_state - original_state)
                )
                if not bool((permutation_error <= 3e-5).item()):
                    raise RuntimeError(
                        "Generator obstacle-set permutation invariance failed: "
                        f"max_abs_error={float(permutation_error.cpu()):.9g}"
                    )

            transform_ids = np.arange(batch_size, dtype=np.int64) % len(
                D4_MATRICES
            )
            transformed_X, _ = augment_d4_batch(
                data["train_X"][:batch_size],
                data["train_y"][:batch_size, 0:4],
                np.random.default_rng(0),
                transform_ids=transform_ids,
            )
            transformed_x = torch.as_tensor(
                norm_X(transformed_X, norm),
                dtype=torch.float32,
                device=device,
            )
            transformed_raw = torch.as_tensor(
                transformed_X, dtype=torch.float32, device=device
            )
            transformed_output = generator(transformed_x, z)
            transformed_state, _ = params_to_state_torch(
                transformed_raw,
                transformed_output,
                tensor_normalizer(norm, device),
            )
            expected_state = original_state.detach().cpu().numpy().copy()
            for row, transform_id in enumerate(transform_ids):
                matrix = D4_MATRICES[int(transform_id)]
                expected_state[row, 0:2] = (
                    (expected_state[row, 0:2] - 0.5) @ matrix.T + 0.5
                )
                expected_state[row, 2:4] = expected_state[row, 2:4] @ matrix.T
            expected_state = torch.as_tensor(
                expected_state, dtype=torch.float32, device=device
            )
            equivariance_error = torch.max(
                torch.abs(transformed_state - expected_state)
            )
            if not bool((equivariance_error <= 5e-5).item()):
                raise RuntimeError(
                    "Generator D4 equivariance failed: "
                    f"max_abs_error={float(equivariance_error.cpu()):.9g}"
                )

        generator.train()
        output = generator(xb, z)
        expected_shape = (batch_size, 4)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Generator smoke-test shape {tuple(output.shape)} != {expected_shape}"
            )
        if not bool(torch.isfinite(output).all().item()):
            raise FloatingPointError("Generator smoke test produced NaN or infinity")
        state, params = params_to_state_torch(
            raw_x, output, tensor_normalizer(norm, device)
        )
        recovered_increment = (
            2.0 * params[:, 0:2] / generator.motion_dt
            - 2.0 * raw_x[:, 2:4]
        )
        reconstructed_params = params_from_control_increment_torch(
            raw_x[:, 2:4],
            recovered_increment,
            generator.motion_dt,
            generator.normalized_speed_limit,
        )
        dynamics_error = torch.max(torch.abs(params - reconstructed_params))
        if not bool((dynamics_error <= 2e-5).item()):
            raise RuntimeError(
                "Generator smoke test violated its exact dynamics manifold: "
                f"max_abs_error={float(dynamics_error.detach().cpu()):.9g}"
            )
        logits = discriminator(xb, output.detach())
        if tuple(state.shape) != expected_shape or tuple(params.shape) != expected_shape:
            raise RuntimeError("State reconstruction smoke-test shape is incorrect")
        if tuple(logits.shape) != (batch_size,):
            raise RuntimeError("Discriminator smoke-test shape is incorrect")
        smoke_loss = output.square().mean() + logits.square().mean()
        if not bool(torch.isfinite(smoke_loss).item()):
            raise FloatingPointError("Network smoke-test loss is not finite")
        generator.zero_grad(set_to_none=True)
        discriminator.zero_grad(set_to_none=True)
        smoke_loss.backward()
        for model_name, model in (
            ("generator", generator),
            ("discriminator", discriminator),
        ):
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(
                bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ):
                raise FloatingPointError(
                    f"{model_name} smoke test produced missing or non-finite gradients"
                )
            if not any(
                bool(torch.any(torch.abs(gradient) > 0.0).item())
                for gradient in gradients
            ):
                raise RuntimeError(
                    f"{model_name} smoke test produced only zero gradients"
                )
    finally:
        torch.set_rng_state(cpu_rng_state)
        if device.type == "cuda" and cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    print(
        "MODEL SMOKE TEST PASSED: generator/discriminator forward shapes, "
        "finite values, exact control-consistent dynamics, obstacle-set "
        "permutation invariance, D4 equivariance, state reconstruction, and "
        "backward gradients",
        flush=True,
    )


def run_warm_start_smoke_test(data, norm, args, device):
    """Verify that the selected source model loads and executes unchanged."""
    if not args.warm_start_checkpoint:
        return
    checkpoint_path = Path(args.warm_start_checkpoint).expanduser()
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
    selected = validate_warm_start_checkpoint(
        checkpoint, args, data["train_X"].shape[1]
    )
    G = LocalMotionGenerator(
        data["train_X"].shape[1], norm, noise_dim=args.noise_dim
    ).to(device)
    D = LocalMotionDiscriminator(data["train_X"].shape[1]).to(device)
    load_generator_state_for_version(
        G,
        checkpoint["best_generator_state_dict"],
        checkpoint["script_version"],
    )
    expected_buffers = {
        "condition_mean": norm["x_mean"],
        "condition_std": norm["x_std"],
        "motion_p_mean": norm["p_mean"],
        "motion_p_std": norm["p_std"],
        "local_control_increment_mean": norm[
            "local_control_increment_mean"
        ],
        "local_control_increment_std": norm[
            "local_control_increment_std"
        ],
        "control_increment_component_limit": norm[
            "control_increment_component_limit"
        ],
        "motion_dt": norm["median_label_dt"],
        "normalized_speed_limit": norm["normalized_speed_limit"],
        "kinematic_prior_strength": norm["kinematic_prior_strength"],
    }
    for buffer_name, expected in expected_buffers.items():
        actual = getattr(G, buffer_name).detach().cpu().numpy()
        if not np.allclose(
            actual,
            np.asarray(expected),
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Warm-start generator normalizer disagrees with the current "
                f"training-only normalizer: {buffer_name}"
            )
    D.load_state_dict(checkpoint["best_discriminator_state_dict"])
    G.eval()
    D.eval()
    count = min(16, len(data["train_X"]))
    xb = torch.as_tensor(
        norm_X(data["train_X"][:count], norm),
        dtype=torch.float32,
        device=device,
    )
    z0 = torch.zeros(count, args.noise_dim, device=device)
    with torch.no_grad():
        prediction = G(xb, z0)
        logits = D(xb, prediction)
    if prediction.shape != (count, 4) or logits.shape != (count,):
        raise RuntimeError("Warm-start model produced unexpected tensor shapes")
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("Warm-start generator produced non-finite values")
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("Warm-start discriminator produced non-finite values")
    print(
        "WARM-START SMOKE TEST PASSED: "
        f"source_epoch={selected['epoch']}, "
        "selected model passes exact compatibility checks and produces finite outputs",
        flush=True,
    )


def direction_cosine_loss(pred_unit, true_unit):
    cos = torch.sum(pred_unit * true_unit, dim=1)
    return torch.mean(1.0 - cos)


def state_threshold_hinge(pred_state, expert_state, threshold, margin_fraction):
    err = torch.norm(pred_state - expert_state, dim=1)
    target = float(threshold) * float(margin_fraction)
    return torch.mean(torch.relu(err - target) ** 2)


def top_tail_state_loss(pred_state, expert_state, tail_fraction):
    err = torch.norm(pred_state - expert_state, dim=1)
    k = max(1, int(len(err) * float(tail_fraction)))
    vals = torch.topk(err, k=k, largest=True).values
    return torch.mean(vals ** 2)


def vector_threshold_hinge(pred, expert, threshold, margin_fraction):
    """Penalize vector errors before they cross a reported accuracy limit."""
    err = torch.norm(pred - expert, dim=1)
    target = float(threshold) * float(margin_fraction)
    excess = torch.relu(err - target)
    # A purely squared hinge has almost no gradient close to the decision
    # boundary, exactly where a 95%-accurate model must correct its remaining
    # tail.  The scaled linear term keeps a useful gradient while the square
    # still penalizes large misses more strongly.
    return torch.mean(excess ** 2 + float(threshold) * excess)


def top_tail_vector_loss(pred, expert, tail_fraction):
    """Directly train the worst vector errors instead of only their mean."""
    err = torch.norm(pred - expert, dim=1)
    k = max(1, int(len(err) * float(tail_fraction)))
    return torch.mean(torch.topk(err, k=k, largest=True).values ** 2)


def make_torch_generator(device, seed):
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def evaluate_generator(G, X, y, norm, device, args, candidate_count, seed):
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    G.eval()
    Xn = norm_X(X, norm)
    expert = y[:, 0:4].astype(np.float32)
    bounds = {
        "step_min": norm["step_min"],
        "step_max": norm["step_max"],
        "speed_min": norm["speed_min"],
        "speed_max": norm["speed_max"],
    }
    generator = make_torch_generator(device, seed)
    best_parts = []
    det_parts = []
    candidate_deviation_sum = 0.0
    candidate_deviation_count = 0
    candidate_deviation_max = 0.0
    with torch.no_grad():
        for start in range(0, len(Xn), args.batch_size):
            stop = min(len(Xn), start + args.batch_size)
            xb = torch.tensor(Xn[start:stop], dtype=torch.float32, device=device)
            raw_x = X[start:stop]
            expert_batch = expert[start:stop]
            z0 = torch.zeros(len(xb), args.noise_dim, device=device)
            p0 = unnorm_params(G(xb, z0).detach().cpu().numpy(), norm)
            det_batch = params_to_state_np(raw_x, p0, bounds=bounds)
            best_batch = det_batch.copy()
            best_err = np.linalg.norm(best_batch - expert_batch, axis=1)

            remaining = candidate_count - 1
            while remaining > 0:
                chunk = min(int(args.eval_candidate_chunk), remaining)
                xb_rep = xb.repeat_interleave(chunk, dim=0)
                z = args.eval_noise_scale * torch.randn(
                    len(xb_rep), args.noise_dim, device=device, generator=generator
                )
                pn = G(xb_rep, z).detach().cpu().numpy()
                params = unnorm_params(pn, norm)
                raw_rep = np.repeat(raw_x, chunk, axis=0)
                candidate_states = params_to_state_np(raw_rep, params, bounds=bounds).reshape(len(xb), chunk, 4)
                deviations = np.linalg.norm(candidate_states - det_batch[:, None, :], axis=2)
                candidate_deviation_sum += float(np.sum(deviations))
                candidate_deviation_count += int(deviations.size)
                candidate_deviation_max = max(candidate_deviation_max, float(np.max(deviations)))
                errors = np.linalg.norm(candidate_states - expert_batch[:, None, :], axis=2)
                local_i = np.argmin(errors, axis=1)
                local_err = errors[np.arange(len(xb)), local_i]
                improve = local_err < best_err
                if np.any(improve):
                    best_batch[improve] = candidate_states[np.arange(len(xb))[improve], local_i[improve]]
                    best_err[improve] = local_err[improve]
                remaining -= chunk
            det_parts.append(det_batch)
            best_parts.append(best_batch)
    det = np.vstack(det_parts).astype(np.float32)
    best = np.vstack(best_parts).astype(np.float32)
    if not np.isfinite(det).all() or not np.isfinite(best).all():
        raise FloatingPointError("Generator evaluation produced NaN or infinity")

    def metric_block(prefix, pred):
        pos_err = np.linalg.norm(pred[:, 0:2] - expert[:, 0:2], axis=1)
        vel_err = np.linalg.norm(pred[:, 2:4] - expert[:, 2:4], axis=1)
        state_err = np.linalg.norm(pred - expert, axis=1)
        predicted_delta = pred[:, 0:2] - X[:, 0:2]
        expert_delta = expert[:, 0:2] - X[:, 0:2]
        predicted_control_increment = control_increment_from_delta_np(
            X[:, 2:4], predicted_delta, norm["median_label_dt"]
        )
        expert_control_increment = control_increment_from_delta_np(
            X[:, 2:4], expert_delta, norm["median_label_dt"]
        )
        control_increment_err = np.linalg.norm(
            predicted_control_increment - expert_control_increment, axis=1
        )
        return {
            f"{prefix}_full_state_accuracy_percent": float(100.0 * np.mean(state_err <= args.state_accuracy_threshold)),
            f"{prefix}_full_state_mse_loss": float(np.mean((pred - expert) ** 2)),
            f"{prefix}_position_accuracy_percent": float(100.0 * np.mean(pos_err <= args.position_accuracy_threshold)),
            f"{prefix}_velocity_accuracy_percent": float(100.0 * np.mean(vel_err <= args.velocity_accuracy_threshold)),
            f"{prefix}_mean_position_error": float(np.mean(pos_err)),
            f"{prefix}_mean_velocity_error": float(np.mean(vel_err)),
            f"{prefix}_mean_full_state_error": float(np.mean(state_err)),
            f"{prefix}_p95_position_error": float(np.percentile(pos_err, 95)),
            f"{prefix}_p95_velocity_error": float(np.percentile(vel_err, 95)),
            f"{prefix}_control_increment_accuracy_at_0_05_percent": float(
                100.0 * np.mean(control_increment_err <= 0.05)
            ),
            f"{prefix}_mean_control_increment_error": float(
                np.mean(control_increment_err)
            ),
            f"{prefix}_p95_control_increment_error": float(
                np.percentile(control_increment_err, 95)
            ),
            f"{prefix}_p90_full_state_error": float(np.percentile(state_err, 90)),
            f"{prefix}_p95_full_state_error": float(np.percentile(state_err, 95)),
            f"{prefix}_full_state_accuracy_at_0_05_percent": float(
                100.0 * np.mean(state_err <= 0.05)
            ),
            f"{prefix}_full_state_accuracy_at_0_10_percent": float(
                100.0 * np.mean(state_err <= 0.10)
            ),
            f"{prefix}_position_accuracy_at_0_005_percent": float(
                100.0 * np.mean(pos_err <= 0.005)
            ),
            f"{prefix}_position_accuracy_at_0_01_percent": float(
                100.0 * np.mean(pos_err <= 0.01)
            ),
            f"{prefix}_position_accuracy_at_0_025_percent": float(
                100.0 * np.mean(pos_err <= 0.025)
            ),
            f"{prefix}_position_accuracy_at_0_05_percent": float(
                100.0 * np.mean(pos_err <= 0.05)
            ),
            f"{prefix}_velocity_accuracy_at_0_025_percent": float(
                100.0 * np.mean(vel_err <= 0.025)
            ),
            f"{prefix}_velocity_accuracy_at_0_05_percent": float(
                100.0 * np.mean(vel_err <= 0.05)
            ),
            f"{prefix}_velocity_accuracy_at_0_10_percent": float(
                100.0 * np.mean(vel_err <= 0.10)
            ),
        }

    out = {}
    out.update(metric_block("generator_bestof", best))
    out.update(metric_block("generator_deterministic", det))
    out["candidate_count"] = int(candidate_count)
    out["mean_random_candidate_deviation_from_deterministic_state"] = float(
        candidate_deviation_sum / max(1, candidate_deviation_count)
    )
    out["maximum_random_candidate_deviation_from_deterministic_state"] = float(candidate_deviation_max)
    out["generator_bestof_is_oracle_metric"] = True
    out["generator_deterministic_is_deployable_single_output_metric"] = True
    return out


def evaluate_adv_discriminator(G, D, X, y, norm, device, args, seed):
    G.eval()
    D.eval()
    Xn = norm_X(X, norm)
    real_pn = norm_params(target_direction_step_speed(X, y), norm)
    total_correct = 0
    total_count = 0
    total_loss = 0.0
    generator = make_torch_generator(device, seed)
    tnorm = tensor_normalizer(norm, device)
    with torch.no_grad():
        for start in range(0, len(Xn), args.batch_size):
            xb = torch.tensor(Xn[start:start + args.batch_size], dtype=torch.float32, device=device)
            real = torch.tensor(real_pn[start:start + args.batch_size], dtype=torch.float32, device=device)
            z = args.train_noise_scale * torch.randn(
                len(xb), args.noise_dim, device=device, generator=generator
            )
            fake_raw = G(xb, z).detach()
            fake, _ = canonical_params_norm_torch(fake_raw, tnorm)
            real_logits = D(xb, real)
            fake_logits = D(xb, fake)
            loss_real = nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            loss = 0.5 * (loss_real + loss_fake)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Adversarial discriminator evaluation produced a non-finite loss"
                )
            real_prob = torch.sigmoid(real_logits)
            fake_prob = torch.sigmoid(fake_logits)
            correct = int((real_prob >= 0.5).sum().item() + (fake_prob < 0.5).sum().item())
            count = len(real_prob) + len(fake_prob)
            total_correct += correct
            total_count += count
            total_loss += float(loss.detach().cpu()) * count
    return {
        "adversarial_discriminator_accuracy_percent": float(100.0 * total_correct / max(1, total_count)),
        "adversarial_discriminator_bce_loss": float(total_loss / max(1, total_count)),
        "adversarial_discriminator_is_generator_accuracy_target": False,
        "adversarial_discriminator_interpretation": (
            "passive GAN-balance diagnostic; approximately 50% is not a failed "
            "motion-prediction accuracy metric"
        ),
    }


def validation_selection_values(generator_metrics):
    """Return the deployable minimum and validation-only lexicographic score."""
    state_accuracy = float(
        generator_metrics[
            "generator_deterministic_full_state_accuracy_percent"
        ]
    )
    position_accuracy = float(
        generator_metrics[
            "generator_deterministic_position_accuracy_percent"
        ]
    )
    velocity_accuracy = float(
        generator_metrics[
            "generator_deterministic_velocity_accuracy_percent"
        ]
    )
    minimum_accuracy = min(
        state_accuracy, position_accuracy, velocity_accuracy
    )
    # Worst deployable accuracy dominates. Remaining terms break exact ties.
    # Oracle best-of and discriminator values are deliberately absent.
    score = (
        1.0e9 * minimum_accuracy
        + 1.0e6 * state_accuracy
        + 1.0e3 * velocity_accuracy
        + position_accuracy
        - 1.0e-2
        * float(generator_metrics["generator_deterministic_mean_full_state_error"])
        - 1.0e-3
        * float(generator_metrics["generator_deterministic_p95_full_state_error"])
    )
    if not math.isfinite(score):
        raise FloatingPointError("Validation selection score is non-finite")
    return float(minimum_accuracy), float(score)


def supervised_learning_rate(base_lr, epoch, total_epochs, decay_start, final_factor):
    """Cosine-decay the supervised learning rate after an initial plateau."""
    if total_epochs <= 1:
        return float(base_lr)
    progress = (float(epoch) - 1.0) / max(1.0, float(total_epochs) - 1.0)
    if progress <= float(decay_start):
        return float(base_lr)
    local = (progress - float(decay_start)) / max(1e-12, 1.0 - float(decay_start))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, local))))
    multiplier = float(final_factor) + (1.0 - float(final_factor)) * cosine
    return float(base_lr) * multiplier


def set_generator_trainability(generator, residual_only):
    """V7.7 always trains its new invariant architecture from scratch."""
    if residual_only:
        raise ValueError(
            "V7.7 has no legacy residual-only phase; train all parameters"
        )
    trainable = 0
    total = 0
    for parameter in generator.parameters():
        parameter.requires_grad_(True)
        total += int(parameter.numel())
        trainable += int(parameter.numel())
    if trainable <= 0 or total <= 0:
        raise RuntimeError("Generator trainability configuration selected no parameters")
    return {
        "mode": "invariant_architecture_all_parameters",
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
    }


def compute_hard_example_probabilities(G, train_X, train_y, norm, device, args):
    """Create training-only sampling probabilities from deterministic errors.

    Every ordinary epoch still contains one complete uniform pass.  These
    probabilities are used only for an additional set of training examples,
    so hard-example mining cannot delete easy/common behavior.  Validation and
    testing labels are never read here.
    """
    G.eval()
    Xn = norm_X(train_X, norm)
    tnorm = tensor_normalizer(norm, device)
    velocity_error_parts = []
    state_error_parts = []
    control_error_parts = []
    motion_dt = float(norm["median_label_dt"])
    if not math.isfinite(motion_dt) or motion_dt <= 0.0:
        raise RuntimeError("Hard-example mining requires a positive motion interval")
    with torch.no_grad():
        for start in range(0, len(Xn), args.batch_size):
            end = min(len(Xn), start + args.batch_size)
            xb = torch.as_tensor(
                Xn[start:end], dtype=torch.float32, device=device
            )
            raw_x = torch.as_tensor(
                train_X[start:end], dtype=torch.float32, device=device
            )
            expert_state = torch.as_tensor(
                train_y[start:end, 0:4], dtype=torch.float32, device=device
            )
            z0 = torch.zeros(len(xb), args.noise_dim, device=device)
            pred_pn = G(xb, z0)
            pred_state, pred_params = params_to_state_torch(
                raw_x, pred_pn, tnorm
            )
            expert_delta = expert_state[:, 0:2] - raw_x[:, 0:2]
            expert_control = (
                2.0 * expert_delta / motion_dt - 2.0 * raw_x[:, 2:4]
            )
            pred_control = (
                2.0 * pred_params[:, 0:2] / motion_dt
                - 2.0 * raw_x[:, 2:4]
            )
            velocity_error_parts.append(
                torch.norm(pred_state[:, 2:4] - expert_state[:, 2:4], dim=1)
                .cpu().numpy()
            )
            state_error_parts.append(
                torch.norm(pred_state - expert_state, dim=1).cpu().numpy()
            )
            control_error_parts.append(
                torch.norm(pred_control - expert_control, dim=1).cpu().numpy()
            )
    velocity_error = np.concatenate(velocity_error_parts).astype(np.float64)
    state_error = np.concatenate(state_error_parts).astype(np.float64)
    control_error = np.concatenate(control_error_parts).astype(np.float64)
    if not (
        np.all(np.isfinite(velocity_error))
        and np.all(np.isfinite(state_error))
        and np.all(np.isfinite(control_error))
    ):
        raise FloatingPointError("Hard-example mining produced non-finite errors")
    strict_ratio = np.maximum.reduce([
        velocity_error / float(args.velocity_accuracy_threshold),
        control_error / float(args.velocity_accuracy_threshold),
        state_error / float(args.state_accuracy_threshold),
    ])
    excess = np.maximum(
        strict_ratio - float(args.hard_priority_start_ratio), 0.0
    )
    excess = np.minimum(excess, float(args.hard_priority_cap))
    priority = 1.0 + float(args.hard_priority_strength) * np.power(
        excess, float(args.hard_priority_power)
    )
    priority = np.maximum(priority, 1e-12)
    probability = priority / np.sum(priority)
    report = {
        "rows": int(len(train_X)),
        "training_velocity_accuracy_percent": float(
            100.0 * np.mean(
                velocity_error <= float(args.velocity_accuracy_threshold)
            )
        ),
        "training_control_increment_accuracy_percent": float(
            100.0 * np.mean(
                control_error <= float(args.velocity_accuracy_threshold)
            )
        ),
        "training_mean_velocity_error": float(np.mean(velocity_error)),
        "training_p95_velocity_error": float(np.percentile(velocity_error, 95)),
        "training_max_sampling_probability_ratio": float(
            np.max(probability) / max(np.min(probability), 1e-18)
        ),
        "extra_hard_examples_per_epoch": int(
            round(len(train_X) * float(args.hard_example_fraction))
        ),
    }
    return probability.astype(np.float64), report


def train_completed_gan(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_X = data["train_X"]
    train_y = data["train_y"]
    val_X = data["validation_X"]
    val_y = data["validation_y"]
    train_params = target_direction_step_speed(train_X, train_y)
    norm = fit_normalizer(
        train_X,
        train_params,
        motion_dt=args.runtime_effective_dt_normalized,
    )
    Xn = norm_X(train_X, norm)
    Pn = norm_params(train_params, norm)
    cond_dim = train_X.shape[1]
    G = LocalMotionGenerator(cond_dim, norm, noise_dim=args.noise_dim).to(device)
    D = LocalMotionDiscriminator(cond_dim).to(device)
    opt_G = torch.optim.AdamW(
        G.parameters(),
        lr=args.lr_g,
        betas=(0.9, 0.99),
        weight_decay=args.generator_weight_decay,
    )
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.95), weight_decay=1e-6)
    tnorm = tensor_normalizer(norm, device)
    rng = np.random.default_rng(args.seed)
    total_epochs = args.supervised_epochs + args.adversarial_epochs
    start_epoch = 1
    best_score = -1e18
    best_G = None
    best_D = None
    history = []
    evaluations_without_improvement = 0
    supervised_early_stop_epoch = None
    restored_best_for_adversarial = False
    hard_sampling_probabilities = None
    hard_sampling_report = None
    last_hard_mining_epoch = None
    previous_trainability_mode = None
    run_config = checkpoint_run_config(args)
    process_training_start = time.time()
    probe_count = min(len(train_X), int(args.generalization_probe_rows))
    generalization_probe_indices = np.linspace(
        0, len(train_X) - 1, num=probe_count, dtype=np.int64
    )

    checkpoint_path = training_checkpoint_path(args)
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume was requested but checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint.get("format") != "v7_rrt_sample_gan_training_checkpoint_v1":
            raise RuntimeError("Unsupported or stale training checkpoint format")
        if checkpoint.get("script_version") != SCRIPT_VERSION:
            raise RuntimeError("Training checkpoint was created by a different script version")
        if checkpoint.get("dataset_sha256") != args.dataset_sha256:
            raise RuntimeError("Checkpoint belongs to a different dataset; refusing unsafe resume")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError(
                "Training checkpoint arguments differ from this command; refusing an ambiguous resume"
            )
        if int(checkpoint.get("cond_dim", -1)) != cond_dim or int(checkpoint.get("noise_dim", -1)) != args.noise_dim:
            raise RuntimeError("Checkpoint architecture does not match this run")
        G.load_state_dict(checkpoint["generator_state_dict"])
        D.load_state_dict(checkpoint["discriminator_state_dict"])
        opt_G.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        opt_D.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        start_epoch = int(checkpoint["last_completed_epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_G = checkpoint["best_generator_state_dict"]
        best_D = checkpoint["best_discriminator_state_dict"]
        history = list(checkpoint.get("history", []))
        evaluations_without_improvement = int(
            checkpoint.get("evaluations_without_improvement", 0)
        )
        supervised_early_stop_epoch = checkpoint.get("supervised_early_stop_epoch")
        restored_best_for_adversarial = bool(
            checkpoint.get("restored_best_for_adversarial", start_epoch > args.supervised_epochs)
        )
        stored_probabilities = checkpoint.get("hard_sampling_probabilities")
        if stored_probabilities is not None:
            hard_sampling_probabilities = np.asarray(
                stored_probabilities, dtype=np.float64
            )
            if hard_sampling_probabilities.shape != (len(Xn),):
                raise RuntimeError(
                    "Checkpoint hard-example probabilities have the wrong shape"
                )
            if not np.all(np.isfinite(hard_sampling_probabilities)):
                raise RuntimeError(
                    "Checkpoint hard-example probabilities are non-finite"
                )
            probability_sum = float(np.sum(hard_sampling_probabilities))
            if probability_sum <= 0.0:
                raise RuntimeError(
                    "Checkpoint hard-example probabilities have zero mass"
                )
            hard_sampling_probabilities /= probability_sum
        hard_sampling_report = checkpoint.get("hard_sampling_report")
        stored_refresh_epoch = checkpoint.get("last_hard_mining_epoch")
        last_hard_mining_epoch = (
            None if stored_refresh_epoch is None else int(stored_refresh_epoch)
        )
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        print(f"RESUMED TRAINING AT EPOCH {start_epoch} FROM {checkpoint_path}", flush=True)
    elif args.warm_start_checkpoint:
        source_path = Path(args.warm_start_checkpoint).expanduser()
        source_checkpoint = torch_load_checkpoint(source_path, map_location=device)
        source_selected = validate_warm_start_checkpoint(
            source_checkpoint, args, cond_dim
        )
        load_generator_state_for_version(
            G,
            source_checkpoint["best_generator_state_dict"],
            source_checkpoint["script_version"],
        )
        D.load_state_dict(source_checkpoint["best_discriminator_state_dict"])
        best_G = {
            key: value.detach().cpu().clone()
            for key, value in G.state_dict().items()
        }
        best_D = {
            key: value.detach().cpu().clone()
            for key, value in D.state_dict().items()
        }
        best_score = float(source_selected["selection_score"])
        warm_row = dict(source_selected)
        warm_row["epoch"] = 0
        warm_row["phase"] = "warm_start_selected_checkpoint"
        warm_row["warm_start_source_epoch"] = int(source_selected["epoch"])
        warm_row["warm_start_source_script_version"] = source_checkpoint[
            "script_version"
        ]
        warm_row["warm_start_source_checkpoint"] = str(source_path.resolve())
        warm_row["generator_learning_rate"] = 0.0
        history = [warm_row]
        print(
            "WARM STARTED NEW ACCURACY EXPERIMENT FROM VALIDATION-SELECTED "
            f"SOURCE EPOCH {source_selected['epoch']}: {source_path}",
            flush=True,
        )
        print(json.dumps(warm_row), flush=True)

    if start_epoch == 1 and not history:
        # The zero-initialized residual makes epoch 0 the exact label-free
        # nominal LQR controller. Preserve it as a selectable baseline so an
        # optimizer step can never silently replace a stronger model.
        baseline_metrics = evaluate_generator(
            G,
            val_X,
            val_y,
            norm,
            device,
            args,
            args.eval_candidates,
            seed=args.seed + 1002,
        )
        baseline_training_probe = evaluate_generator(
            G,
            train_X[generalization_probe_indices],
            train_y[generalization_probe_indices],
            norm,
            device,
            args,
            candidate_count=1,
            seed=args.seed + 1003,
        )
        baseline_discriminator = evaluate_adv_discriminator(
            G, D, val_X, val_y, norm, device, args, seed=args.seed + 202
        )
        baseline_minimum, baseline_score = validation_selection_values(
            baseline_metrics
        )
        baseline_trainability = set_generator_trainability(
            G, residual_only=False
        )
        baseline_row = {
            "epoch": 0,
            "phase": "nominal_lqr_baseline",
            "generator_learning_rate": 0.0,
            "generator_trainability_mode": baseline_trainability["mode"],
            "generator_trainable_parameters": baseline_trainability[
                "trainable_parameters"
            ],
            "train_g_loss": 0.0,
            "train_d_loss": 0.0,
            "training_generalization_probe_rows": int(probe_count),
            "training_probe_deterministic_position_accuracy_percent": float(
                baseline_training_probe[
                    "generator_deterministic_position_accuracy_percent"
                ]
            ),
            "training_probe_deterministic_velocity_accuracy_percent": float(
                baseline_training_probe[
                    "generator_deterministic_velocity_accuracy_percent"
                ]
            ),
            "training_probe_deterministic_full_state_accuracy_percent": float(
                baseline_training_probe[
                    "generator_deterministic_full_state_accuracy_percent"
                ]
            ),
            "training_probe_deterministic_mean_velocity_error": float(
                baseline_training_probe[
                    "generator_deterministic_mean_velocity_error"
                ]
            ),
            **baseline_metrics,
            **baseline_discriminator,
            "training_validation_velocity_accuracy_gap_percent": float(
                baseline_training_probe[
                    "generator_deterministic_velocity_accuracy_percent"
                ]
                - baseline_metrics[
                    "generator_deterministic_velocity_accuracy_percent"
                ]
            ),
            "selection_score": baseline_score,
            "minimum_deployable_validation_accuracy_percent": baseline_minimum,
            "deployable_validation_target_reached": bool(
                baseline_minimum >= args.target_accuracy
            ),
        }
        history = [baseline_row]
        best_score = baseline_score
        best_G = {
            key: value.detach().cpu().clone()
            for key, value in G.state_dict().items()
        }
        best_D = {
            key: value.detach().cpu().clone()
            for key, value in D.state_dict().items()
        }
        print("EPOCH-0 NOMINAL LQR VALIDATION BASELINE", flush=True)
        print(json.dumps(baseline_row), flush=True)

    for epoch in range(start_epoch, total_epochs + 1):
        if (
            supervised_early_stop_epoch is not None
            and epoch <= args.supervised_epochs
        ):
            # Preserve global epoch numbering/checkpoint compatibility while
            # skipping supervised epochs that validation proved unnecessary.
            continue
        use_adv = epoch > args.supervised_epochs
        if use_adv and not restored_best_for_adversarial:
            # Adversarial fine-tuning starts from the best validation-selected
            # supervised model, never from an overfit final supervised epoch.
            if best_G is not None:
                G.load_state_dict(best_G)
            if best_D is not None:
                D.load_state_dict(best_D)
            opt_G = torch.optim.AdamW(
                G.parameters(), lr=args.lr_g * args.adversarial_lr_factor,
                betas=(0.5, 0.95),
                weight_decay=args.generator_weight_decay,
            )
            opt_D = torch.optim.AdamW(
                D.parameters(), lr=args.lr_d, betas=(0.5, 0.95), weight_decay=1e-6
            )
            restored_best_for_adversarial = True
            print("RESTORED BEST SUPERVISED CHECKPOINT FOR ADVERSARIAL PHASE", flush=True)
        trainability = set_generator_trainability(G, residual_only=False)
        if trainability["mode"] != previous_trainability_mode:
            print(
                "GENERATOR FINE-TUNING PHASE: " + json.dumps(trainability),
                flush=True,
            )
            previous_trainability_mode = trainability["mode"]
        if not use_adv:
            scheduled_lr = supervised_learning_rate(
                args.lr_g,
                epoch,
                max(1, args.supervised_epochs),
                args.lr_decay_start_fraction,
                args.lr_final_factor,
            )
            for parameter_group in opt_G.param_groups:
                parameter_group["lr"] = scheduled_lr
        if not use_adv and args.hard_example_fraction > 0.0:
            refresh_due = (
                hard_sampling_probabilities is None
                or last_hard_mining_epoch is None
                or epoch - last_hard_mining_epoch
                >= args.hard_mining_refresh_epochs
            )
            if refresh_due:
                hard_sampling_probabilities, hard_sampling_report = (
                    compute_hard_example_probabilities(
                        G, train_X, train_y, norm, device, args
                    )
                )
                last_hard_mining_epoch = int(epoch)
                print(
                    "HARD-EXAMPLE MINING REFRESH: "
                    + json.dumps(hard_sampling_report),
                    flush=True,
                )
            extra_count = int(
                round(len(Xn) * float(args.hard_example_fraction))
            )
            extra_hard = rng.choice(
                len(Xn),
                size=extra_count,
                replace=True,
                p=hard_sampling_probabilities,
            )
            # Preserve one complete uniform pass, then add targeted examples.
            order = np.concatenate([
                rng.permutation(len(Xn)),
                extra_hard,
            ])
            rng.shuffle(order)
        else:
            order = rng.permutation(len(Xn))
        G.train()
        D.train()
        g_losses = []
        d_losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch_X, batch_y = augment_d4_batch(
                train_X[idx],
                train_y[idx, 0:4],
                rng,
                probability=args.d4_augmentation_probability,
            )
            batch_params = target_direction_step_speed(batch_X, batch_y)
            xb = torch.tensor(
                norm_X(batch_X, norm), dtype=torch.float32, device=device
            )
            pb = torch.tensor(
                norm_params(batch_params, norm),
                dtype=torch.float32,
                device=device,
            )
            raw_x = torch.tensor(
                batch_X, dtype=torch.float32, device=device
            )
            expert_state = torch.tensor(
                batch_y, dtype=torch.float32, device=device
            )
            expert_params = torch.tensor(
                batch_params, dtype=torch.float32, device=device
            )

            if use_adv:
                for _ in range(args.d_steps):
                    z = args.train_noise_scale * torch.randn(len(xb), args.noise_dim, device=device)
                    with torch.no_grad():
                        fake_raw_pn = G(xb, z).detach()
                        fake_pn, _ = canonical_params_norm_torch(fake_raw_pn, tnorm)
                    real_logits = D(xb, pb)
                    fake_logits = D(xb, fake_pn)
                    d_loss_real = nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                    d_loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)
                    if not bool(torch.isfinite(d_loss).item()):
                        raise FloatingPointError(
                            f"Non-finite adversarial discriminator loss at epoch {epoch}"
                        )
                    opt_D.zero_grad(set_to_none=True)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                    opt_D.step()
                    d_losses.append(float(d_loss.detach().cpu()))
            else:
                d_losses.append(0.0)

            # The deployable output is always trained explicitly at zero noise.
            # Random-noise samples receive adversarial and light supervised
            # training separately, so GAN fine-tuning cannot silently degrade
            # the deterministic controller.
            z0 = torch.zeros(len(xb), args.noise_dim, device=device)
            pred_pn = G(xb, z0)
            pred_state, pred_params = params_to_state_torch(raw_x, pred_pn, tnorm)
            target_delta = expert_params[:, 0:2]
            target_velocity = expert_params[:, 2:4]
            pred_delta = pred_params[:, 0:2]
            pred_velocity = pred_params[:, 2:4]
            target_control_increment = (
                2.0 * target_delta / tnorm["median_label_dt"]
                - 2.0 * raw_x[:, 2:4]
            )
            pred_control_increment = (
                2.0 * pred_delta / tnorm["median_label_dt"]
                - 2.0 * raw_x[:, 2:4]
            )
            target_control_increment_norm = (
                target_control_increment - tnorm["control_increment_mean"]
            ) / tnorm["control_increment_std"]
            pred_control_increment_norm = (
                pred_control_increment - tnorm["control_increment_mean"]
            ) / tnorm["control_increment_std"]
            target_unit = target_delta / (torch.norm(target_delta, dim=1, keepdim=True) + 1e-8)
            pred_unit = pred_delta / (torch.norm(pred_delta, dim=1, keepdim=True) + 1e-8)
            target_step = torch.norm(target_delta, dim=1, keepdim=True)
            pred_step = torch.norm(pred_delta, dim=1, keepdim=True)

            param_mse = nn.functional.mse_loss(pred_pn, pb)
            state_mse = nn.functional.mse_loss(pred_state, expert_state)
            state_huber = nn.functional.smooth_l1_loss(pred_state, expert_state)
            dir_loss = direction_cosine_loss(pred_unit, target_unit)
            step_mse = nn.functional.mse_loss(pred_step, target_step)
            # These terms directly optimize the previously weak velocity metric:
            # full two-component velocity, direction, magnitude, the 0.05
            # boundary, and the worst-error tail. The primary learned quantity
            # is the control increment; velocity is its exact dynamics output.
            velocity_mse = nn.functional.mse_loss(pred_velocity, target_velocity)
            velocity_huber = nn.functional.smooth_l1_loss(pred_velocity, target_velocity)
            target_velocity_unit = target_velocity / (
                torch.norm(target_velocity, dim=1, keepdim=True) + 1e-8
            )
            pred_velocity_unit = pred_velocity / (
                torch.norm(pred_velocity, dim=1, keepdim=True) + 1e-8
            )
            velocity_direction = direction_cosine_loss(
                pred_velocity_unit, target_velocity_unit
            )
            target_speed = torch.norm(target_velocity, dim=1, keepdim=True)
            pred_speed = torch.norm(pred_velocity, dim=1, keepdim=True)
            velocity_magnitude = nn.functional.mse_loss(pred_speed, target_speed)
            velocity_hinge = vector_threshold_hinge(
                pred_velocity, target_velocity,
                args.velocity_accuracy_threshold,
                args.velocity_threshold_margin_fraction,
            )
            velocity_tail = top_tail_vector_loss(
                pred_velocity, target_velocity, args.velocity_tail_fraction
            )
            control_increment_mse = nn.functional.mse_loss(
                pred_control_increment_norm,
                target_control_increment_norm,
            )
            control_increment_huber = nn.functional.smooth_l1_loss(
                pred_control_increment_norm,
                target_control_increment_norm,
            )
            control_increment_hinge = vector_threshold_hinge(
                pred_control_increment,
                target_control_increment,
                args.velocity_accuracy_threshold,
                args.velocity_threshold_margin_fraction,
            )
            control_increment_tail = top_tail_vector_loss(
                pred_control_increment,
                target_control_increment,
                args.velocity_tail_fraction,
            )
            position_hinge = vector_threshold_hinge(
                pred_state[:, 0:2], expert_state[:, 0:2],
                args.position_accuracy_threshold,
                args.position_threshold_margin_fraction,
            )
            dynamics_consistent_velocity = dynamics_velocity_from_delta_torch(
                raw_x[:, 2:4],
                pred_delta,
                tnorm["median_label_dt"],
                tnorm["normalized_speed_limit"],
            )
            kinematic_consistency = nn.functional.smooth_l1_loss(
                pred_velocity, dynamics_consistent_velocity
            )
            target_xy = raw_x[:, 4:6]
            before_distance = torch.norm(target_xy - raw_x[:, 0:2], dim=1)
            expert_progress = before_distance - torch.norm(target_xy - expert_state[:, 0:2], dim=1)
            predicted_progress = before_distance - torch.norm(target_xy - pred_state[:, 0:2], dim=1)
            progress_mse = nn.functional.mse_loss(predicted_progress, expert_progress)
            hinge = state_threshold_hinge(pred_state, expert_state, args.state_accuracy_threshold, args.state_threshold_margin_fraction)
            tail = top_tail_state_loss(pred_state, expert_state, args.tail_fraction)
            adv_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            stochastic_supervision = torch.tensor(0.0, dtype=torch.float32, device=device)
            if use_adv:
                z_adv = args.train_noise_scale * torch.randn(
                    len(xb), args.noise_dim, device=device
                )
                stochastic_pn = G(xb, z_adv)
                stochastic_state, stochastic_params = params_to_state_torch(
                    raw_x, stochastic_pn, tnorm
                )
                canonical_pred_pn = (
                    stochastic_params - tnorm["p_mean"]
                ) / tnorm["p_std"]
                fake_logits = D(xb, canonical_pred_pn)
                adv_loss = nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                stochastic_supervision = nn.functional.smooth_l1_loss(
                    stochastic_state, expert_state
                )
            g_loss = (
                args.param_mse_weight * param_mse
                + args.state_mse_weight * state_mse
                + args.state_huber_weight * state_huber
                + args.direction_weight * dir_loss
                + args.step_weight * step_mse
                + args.speed_weight * velocity_mse
                + args.velocity_huber_weight * velocity_huber
                + args.velocity_direction_weight * velocity_direction
                + args.velocity_magnitude_weight * velocity_magnitude
                + args.velocity_threshold_weight * velocity_hinge
                + args.velocity_tail_weight * velocity_tail
                + args.control_increment_mse_weight * control_increment_mse
                + args.control_increment_huber_weight * control_increment_huber
                + args.control_increment_threshold_weight
                * control_increment_hinge
                + args.control_increment_tail_weight * control_increment_tail
                + args.position_threshold_weight * position_hinge
                + args.kinematic_consistency_weight
                * tnorm["kinematic_prior_strength"] * kinematic_consistency
                + args.progress_weight * progress_mse
                + args.state_threshold_weight * hinge
                + args.tail_state_weight * tail
                + args.adv_weight * adv_loss
                + args.stochastic_supervision_weight * stochastic_supervision
            )
            if not bool(torch.isfinite(g_loss).item()):
                raise FloatingPointError(
                    f"Non-finite generator loss at epoch {epoch}"
                )
            opt_G.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()
            g_losses.append(float(g_loss.detach().cpu()))

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == total_epochs:
            gm = evaluate_generator(
                G, val_X, val_y, norm, device, args, args.eval_candidates,
                seed=args.seed + 1002,
            )
            training_probe = evaluate_generator(
                G,
                train_X[generalization_probe_indices],
                train_y[generalization_probe_indices],
                norm,
                device,
                args,
                candidate_count=1,
                seed=args.seed + 1003,
            )
            dm = evaluate_adv_discriminator(
                G, D, val_X, val_y, norm, device, args, seed=args.seed + 202
            )
            deterministic_state_accuracy = gm[
                "generator_deterministic_full_state_accuracy_percent"
            ]
            deterministic_position_accuracy = gm[
                "generator_deterministic_position_accuracy_percent"
            ]
            deterministic_velocity_accuracy = gm[
                "generator_deterministic_velocity_accuracy_percent"
            ]
            minimum_deployable_accuracy, score = validation_selection_values(gm)
            row = {
                "epoch": epoch,
                "phase": "adversarial" if use_adv else "supervised",
                "generator_learning_rate": float(opt_G.param_groups[0]["lr"]),
                "generator_trainability_mode": trainability["mode"],
                "generator_trainable_parameters": trainability[
                    "trainable_parameters"
                ],
                "train_g_loss": float(np.mean(g_losses)),
                "train_d_loss": float(np.mean(d_losses)),
                "training_generalization_probe_rows": int(probe_count),
                "training_probe_deterministic_position_accuracy_percent": float(
                    training_probe[
                        "generator_deterministic_position_accuracy_percent"
                    ]
                ),
                "training_probe_deterministic_velocity_accuracy_percent": float(
                    training_probe[
                        "generator_deterministic_velocity_accuracy_percent"
                    ]
                ),
                "training_probe_deterministic_full_state_accuracy_percent": float(
                    training_probe[
                        "generator_deterministic_full_state_accuracy_percent"
                    ]
                ),
                "training_probe_deterministic_mean_velocity_error": float(
                    training_probe["generator_deterministic_mean_velocity_error"]
                ),
                **gm,
                **dm,
                "training_validation_velocity_accuracy_gap_percent": float(
                    training_probe[
                        "generator_deterministic_velocity_accuracy_percent"
                    ] - deterministic_velocity_accuracy
                ),
                "selection_score": float(score),
                "minimum_deployable_validation_accuracy_percent": float(
                    minimum_deployable_accuracy
                ),
                "deployable_validation_target_reached": bool(
                    minimum_deployable_accuracy >= args.target_accuracy
                ),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            elapsed = time.time() - process_training_start
            completed_this_process = max(1, epoch - start_epoch + 1)
            seconds_per_epoch = elapsed / completed_this_process
            eta_seconds = max(0.0, (total_epochs - epoch) * seconds_per_epoch)
            print(
                f"TRAINING PROGRESS: epoch {epoch}/{total_epochs} "
                f"({100.0 * epoch / total_epochs:.2f}%) | "
                f"elapsed={elapsed / 60.0:.2f} min | "
                f"approx_eta={eta_seconds / 60.0:.2f} min",
                flush=True,
            )
            improved = score > best_score + args.early_stopping_min_delta
            if improved:
                best_score = score
                best_G = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
                best_D = {k: v.detach().cpu().clone() for k, v in D.state_dict().items()}
                evaluations_without_improvement = 0
            elif not use_adv:
                evaluations_without_improvement += 1
            if (
                not use_adv
                and args.early_stopping_patience_evaluations > 0
                and epoch >= args.minimum_supervised_epochs
                and minimum_deployable_accuracy >= args.target_accuracy
                and evaluations_without_improvement
                >= args.early_stopping_patience_evaluations
            ):
                supervised_early_stop_epoch = int(epoch)
                print(
                    "SUPERVISED EARLY STOP: "
                    f"epoch={epoch}, validation evaluations without improvement="
                    f"{evaluations_without_improvement}; all deployable validation metrics "
                    f"are at least {args.target_accuracy:g}%. Remaining supervised epochs are skipped.",
                    flush=True,
                )
            atomic_torch_save(
                {
                    "format": "v7_rrt_sample_gan_training_checkpoint_v1",
                    "script_version": SCRIPT_VERSION,
                    "dataset_sha256": args.dataset_sha256,
                    "run_config": run_config,
                    "cond_dim": cond_dim,
                    "noise_dim": args.noise_dim,
                    "last_completed_epoch": epoch,
                    "total_requested_epochs": total_epochs,
                    "generator_state_dict": {k: v.detach().cpu() for k, v in G.state_dict().items()},
                    "discriminator_state_dict": {k: v.detach().cpu() for k, v in D.state_dict().items()},
                    "generator_optimizer_state_dict": opt_G.state_dict(),
                    "discriminator_optimizer_state_dict": opt_D.state_dict(),
                    "best_score": best_score,
                    "best_generator_state_dict": best_G,
                    "best_discriminator_state_dict": best_D,
                    "history": history,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "supervised_early_stop_epoch": supervised_early_stop_epoch,
                    "restored_best_for_adversarial": restored_best_for_adversarial,
                    "hard_sampling_probabilities": hard_sampling_probabilities,
                    "hard_sampling_report": hard_sampling_report,
                    "last_hard_mining_epoch": last_hard_mining_epoch,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
                checkpoint_path,
            )
            print(f"CHECKPOINT SAVED: {checkpoint_path}", flush=True)

    if best_G is not None:
        G.load_state_dict(best_G)
    if best_D is not None:
        D.load_state_dict(best_D)
    G.to(device)
    D.to(device)
    return G, D, norm, device, history


def train_final_discriminator(G, data, norm, device, args):
    print("TRAINING FINAL DISCRIMINATOR CLASSIFIER", flush=True)
    print("Generator is frozen. The classifier is fit only on the training split.", flush=True)
    C = LocalMotionDiscriminator(data["train_X"].shape[1]).to(device)
    opt = torch.optim.AdamW(C.parameters(), lr=args.final_d_lr, betas=(0.5, 0.95), weight_decay=1e-6)
    train_X = data["train_X"]
    train_y = data["train_y"]
    Xn = norm_X(train_X, norm)
    real_pn = norm_params(target_direction_step_speed(train_X, train_y), norm)
    rng = np.random.default_rng(args.seed + 999)
    best_state = None
    best_report = None
    best_selection_accuracy = -math.inf
    best_selection_loss = math.inf
    rows = []
    G.eval()
    tnorm = tensor_normalizer(norm, device)
    generator_sha256 = model_state_sha256(G)
    checkpoint_path = final_discriminator_checkpoint_path(args)
    start_step = 1
    process_classifier_start = time.time()
    run_config = checkpoint_run_config(args)
    if args.resume and checkpoint_path.exists():
        checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
        if checkpoint.get("format") != "v7_rrt_sample_gan_final_discriminator_checkpoint_v1":
            raise RuntimeError("Unsupported or stale final discriminator checkpoint format")
        if checkpoint.get("script_version") != SCRIPT_VERSION:
            raise RuntimeError("Final discriminator checkpoint was created by a different script version")
        if checkpoint.get("dataset_sha256") != args.dataset_sha256:
            raise RuntimeError("Final discriminator checkpoint belongs to a different dataset")
        if checkpoint.get("generator_sha256") != generator_sha256:
            raise RuntimeError("Final discriminator checkpoint belongs to a different generator")
        if checkpoint.get("run_config") != run_config:
            raise RuntimeError(
                "Final discriminator checkpoint arguments differ from this command; refusing an ambiguous resume"
            )
        C.load_state_dict(checkpoint["classifier_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["last_completed_step"]) + 1
        best_state = checkpoint["best_classifier_state_dict"]
        best_report = checkpoint["best_report"]
        best_selection_accuracy = float(checkpoint["best_selection_accuracy"])
        best_selection_loss = float(checkpoint["best_selection_loss"])
        rows = list(checkpoint.get("rows", []))
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        if best_report is not None and min(
            best_report["training_accuracy"], best_report["validation_accuracy"]
        ) >= args.target_accuracy:
            start_step = args.final_d_steps + 1
        print(f"RESUMED FINAL DISCRIMINATOR AT STEP {start_step} FROM {checkpoint_path}", flush=True)
    for step in range(start_step, args.final_d_steps + 1):
        C.train()
        idx = rng.integers(0, len(Xn), size=args.batch_size)
        xb = torch.tensor(Xn[idx], dtype=torch.float32, device=device)
        real = torch.tensor(real_pn[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            z = args.final_d_fake_noise_scale * torch.randn(len(xb), args.noise_dim, device=device)
            zero_count = int(round(len(xb) * args.final_d_zero_noise_fraction))
            if zero_count > 0:
                z[:zero_count] = 0.0
            fake_raw = G(xb, z).detach()
            fake, _ = canonical_params_norm_torch(fake_raw, tnorm)
        real_logits = C(xb, real)
        fake_logits = C(xb, fake)
        loss_real = nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        loss = 0.5 * (loss_real + loss_fake)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"Non-finite final discriminator loss at optimizer step {step}"
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(C.parameters(), 5.0)
        opt.step()
        if step == 1 or step % args.final_d_eval_every == 0 or step == args.final_d_steps:
            tr = evaluate_final_discriminator(C, G, data["train_X"], data["train_y"], norm, device, args, limit=20000, seed=args.seed + 2001)
            va = evaluate_final_discriminator(C, G, data["validation_X"], data["validation_y"], norm, device, args, limit=None, seed=args.seed + 2002)
            report = {
                "step": step,
                "training_accuracy": tr["final_discriminator_accuracy_percent"],
                "training_loss": tr["final_discriminator_bce_loss"],
                "validation_accuracy": va["final_discriminator_accuracy_percent"],
                "validation_loss": va["final_discriminator_bce_loss"],
                "testing_accuracy": None,
                "testing_loss": None,
            }
            rows.append(report)
            print(json.dumps(report), flush=True)
            elapsed = time.time() - process_classifier_start
            completed_this_process = max(1, step - start_step + 1)
            seconds_per_step = elapsed / completed_this_process
            eta_seconds = max(0.0, (args.final_d_steps - step) * seconds_per_step)
            print(
                f"FINAL CLASSIFIER PROGRESS: step {step}/{args.final_d_steps} "
                f"({100.0 * step / args.final_d_steps:.2f}%) | "
                f"elapsed={elapsed / 60.0:.2f} min | "
                f"approx_eta={eta_seconds / 60.0:.2f} min",
                flush=True,
            )
            # The testing split is deliberately excluded from checkpoint
            # selection. It is evaluated once, after the checkpoint is fixed.
            worst_acc = min(report["training_accuracy"], report["validation_accuracy"])
            worst_loss = max(report["training_loss"], report["validation_loss"])
            # Accuracy is the requested classifier metric. Compare it first
            # and use BCE only for an exact accuracy tie. This lexicographic
            # rule mathematically prevents a lower-accuracy checkpoint from
            # replacing a higher-accuracy one, regardless of loss magnitude.
            accuracy_improved = worst_acc > best_selection_accuracy
            exact_accuracy_tie = math.isclose(
                worst_acc, best_selection_accuracy, rel_tol=0.0, abs_tol=1e-12
            )
            loss_improved_on_tie = (
                exact_accuracy_tie and worst_loss < best_selection_loss
            )
            if accuracy_improved or loss_improved_on_tie:
                best_selection_accuracy = float(worst_acc)
                best_selection_loss = float(worst_loss)
                best_report = report
                best_state = {k: v.detach().cpu().clone() for k, v in C.state_dict().items()}
            atomic_torch_save(
                {
                    "format": "v7_rrt_sample_gan_final_discriminator_checkpoint_v1",
                    "script_version": SCRIPT_VERSION,
                    "dataset_sha256": args.dataset_sha256,
                    "generator_sha256": generator_sha256,
                    "run_config": run_config,
                    "last_completed_step": step,
                    "classifier_state_dict": {k: v.detach().cpu() for k, v in C.state_dict().items()},
                    "optimizer_state_dict": opt.state_dict(),
                    "best_classifier_state_dict": best_state,
                    "best_report": best_report,
                    "best_selection_accuracy": best_selection_accuracy,
                    "best_selection_loss": best_selection_loss,
                    "rows": rows,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                },
                checkpoint_path,
            )
            print(f"FINAL DISCRIMINATOR CHECKPOINT SAVED: {checkpoint_path}", flush=True)
            if best_report is None:
                raise RuntimeError(
                    "Final discriminator checkpoint selection did not produce a report"
                )
            selected_worst_accuracy = min(
                best_report["training_accuracy"],
                best_report["validation_accuracy"],
            )
            if selected_worst_accuracy >= args.target_accuracy:
                print("Final discriminator target reached.", flush=True)
                break
    if best_state is not None:
        C.load_state_dict(best_state)
        C.to(device)
    held_out_test = evaluate_final_discriminator(
        C, G, data["testing_X"], data["testing_y"], norm, device, args,
        limit=None, seed=args.seed + 2003,
    )
    best_report = dict(best_report or {})
    best_report["testing_accuracy"] = held_out_test["final_discriminator_accuracy_percent"]
    best_report["testing_loss"] = held_out_test["final_discriminator_bce_loss"]
    best_report["selection_used_testing_split"] = False
    print("HELD-OUT FINAL DISCRIMINATOR TEST:", json.dumps(held_out_test), flush=True)
    return C, best_report, rows


def evaluate_final_discriminator(C, G, X, y, norm, device, args, limit=None, seed=0):
    if limit is not None and len(X) > limit:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=limit, replace=False)
        X = X[idx]
        y = y[idx]
    C.eval()
    G.eval()
    Xn = norm_X(X, norm)
    real_pn = norm_params(target_direction_step_speed(X, y), norm)
    total_correct = 0
    total_count = 0
    total_loss = 0.0
    generator = make_torch_generator(device, seed)
    tnorm = tensor_normalizer(norm, device)
    with torch.no_grad():
        for start in range(0, len(Xn), args.batch_size):
            xb = torch.tensor(Xn[start:start + args.batch_size], dtype=torch.float32, device=device)
            real = torch.tensor(real_pn[start:start + args.batch_size], dtype=torch.float32, device=device)
            z = args.final_d_fake_noise_scale * torch.randn(
                len(xb), args.noise_dim, device=device, generator=generator
            )
            zero_count = int(round(len(xb) * args.final_d_zero_noise_fraction))
            if zero_count > 0:
                z[:zero_count] = 0.0
            fake_raw = G(xb, z).detach()
            fake, _ = canonical_params_norm_torch(fake_raw, tnorm)
            real_logits = C(xb, real)
            fake_logits = C(xb, fake)
            loss_real = nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            loss = 0.5 * (loss_real + loss_fake)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    "Final discriminator evaluation produced a non-finite loss"
                )
            real_prob = torch.sigmoid(real_logits)
            fake_prob = torch.sigmoid(fake_logits)
            correct = int((real_prob >= 0.5).sum().item() + (fake_prob < 0.5).sum().item())
            count = len(real_prob) + len(fake_prob)
            total_correct += correct
            total_count += count
            total_loss += float(loss.detach().cpu()) * count
    return {
        "final_discriminator_accuracy_percent": float(100.0 * total_correct / max(1, total_count)),
        "final_discriminator_bce_loss": float(total_loss / max(1, total_count)),
    }


def parse_obstacles(cond):
    """Decode absolute normalized obstacle centers from a strict V7 condition."""
    obs = []
    if len(cond) <= 6:
        return obs
    rest = np.asarray(cond[6:], dtype=float)
    for i in range(len(rest) // 3):
        ox, oy, rr = rest[3 * i:3 * i + 3]
        if np.isfinite(ox) and np.isfinite(oy) and np.isfinite(rr):
            if 0.0 <= ox <= 1.0 and 0.0 <= oy <= 1.0 and 0.0 < rr <= 0.5:
                obs.append((float(ox), float(oy), float(rr)))
    return obs


def point_segment_distance(c, a, b):
    c = np.asarray(c, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(c - a))
    t = float(np.clip(np.dot(c - a, ab) / denom, 0.0, 1.0))
    p = a + t * ab
    return float(np.linalg.norm(c - p))


def min_clearance(pos, obstacles):
    if not obstacles:
        return 1.0
    vals = [float(np.linalg.norm(np.asarray(pos) - np.asarray([ox, oy])) - rr) for ox, oy, rr in obstacles]
    return float(min(vals))


def segment_hits_obstacle(a, b, obstacles, margin):
    for ox, oy, rr in obstacles:
        if point_segment_distance((ox, oy), a, b) <= rr + margin:
            return True
    return False


def edge_is_invalid(
    a, b, obstacles, margin, workspace_min=0.0, workspace_max=1.0
):
    """Check a straight local edge in configuration space.

    The obstacle and workspace-boundary clearances use the same combined robot
    radius plus safety margin recorded by the expert dataset.
    """
    if point_is_invalid(
        a, obstacles, margin, workspace_min, workspace_max
    ) or point_is_invalid(
        b, obstacles, margin, workspace_min, workspace_max
    ):
        return True
    return segment_hits_obstacle(a, b, obstacles, margin)


def angle_between(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.acos(c))


def rotate_toward(v_from, v_to, max_angle):
    v_from = np.asarray(v_from, dtype=float)
    v_to = np.asarray(v_to, dtype=float)
    if np.linalg.norm(v_from) < 1e-8 or np.linalg.norm(v_to) < 1e-8:
        return v_to
    ang_from = math.atan2(v_from[1], v_from[0])
    ang_to = math.atan2(v_to[1], v_to[0])
    diff = (ang_to - ang_from + math.pi) % (2.0 * math.pi) - math.pi
    diff = float(np.clip(diff, -max_angle, max_angle))
    mag = np.linalg.norm(v_to)
    new_ang = ang_from + diff
    return mag * np.array([math.cos(new_ang), math.sin(new_ang)], dtype=float)


def state_from_direction(parent_state, direction, step_len, speed):
    direction = np.asarray(direction, dtype=np.float32)
    n = float(np.linalg.norm(direction))
    if n < 1e-8:
        return None
    u = direction / n
    out = np.zeros(4, dtype=np.float32)
    out[0:2] = parent_state[0:2] + u * float(step_len)
    out[2:4] = u * float(speed)
    return out


def steer(parent_state, candidate_state, args):
    parent = np.asarray(parent_state, dtype=np.float32)
    cand = np.asarray(candidate_state, dtype=np.float32)
    if parent.shape != (4,) or cand.shape != (4,):
        return None
    if not np.isfinite(parent).all() or not np.isfinite(cand).all():
        return None
    disp = cand[0:2] - parent[0:2]
    dist = float(np.linalg.norm(disp))
    if dist < 1e-9:
        return None
    if dist > args.max_step:
        disp = disp / dist * args.max_step
        dist = args.max_step
    if dist < args.min_step:
        return None
    prev_v = parent[2:4]
    if np.linalg.norm(prev_v) > 1e-6:
        disp = rotate_toward(prev_v, disp, args.max_turn_rad)
    candidate_velocity = cand[2:4].astype(np.float32)
    if np.linalg.norm(candidate_velocity) < 1e-8:
        candidate_velocity = disp.copy()
    if np.linalg.norm(prev_v) > 1e-6:
        candidate_velocity = rotate_toward(prev_v, candidate_velocity, args.max_turn_rad)
    speed = float(np.linalg.norm(candidate_velocity))
    speed = float(np.clip(speed, args.min_speed, args.max_speed))
    velocity_direction = candidate_velocity / max(np.linalg.norm(candidate_velocity), 1e-8)
    out = np.zeros(4, dtype=np.float32)
    out[0:2] = parent[0:2] + disp
    out[2:4] = velocity_direction * speed
    return out


@dataclass
class Node:
    state: np.ndarray
    parent: int
    cost: float
    depth: int
    expansions: int = 0
    source: str = "root"


def condition_from_node(base_cond, state, sample_xy):
    cond = np.asarray(base_cond, dtype=np.float32).copy()
    cond[0:4] = state
    cond[4:6] = np.asarray(sample_xy, dtype=np.float32)
    obstacles = parse_obstacles(base_cond)
    slots = (len(cond) - 6) // 3
    ordered = sorted(
        obstacles,
        key=lambda ob: (
            float(np.linalg.norm(np.asarray(state[0:2]) - np.asarray(ob[0:2])) - ob[2]),
            ob[0], ob[1], ob[2],
        ),
    )
    cond[6:] = 0.0
    for i, (ox, oy, rr) in enumerate(ordered[:slots]):
        j = 6 + 3 * i
        cond[j:j + 3] = (ox, oy, rr)
    return cond


def params_from_states(cond, states):
    dpos = states[:, 0:2] - cond[0:2][None, :]
    return np.concatenate([dpos, states[:, 2:4]], axis=1).astype(np.float32)


def discriminator_prob(D, cond, states, norm, device):
    params = params_from_states(cond, states)
    pn = norm_params(params, norm)
    cond_n = norm_X(np.repeat(cond[None, :], len(states), axis=0), norm)
    with torch.no_grad():
        xb = torch.tensor(cond_n, dtype=torch.float32, device=device)
        pb = torch.tensor(pn, dtype=torch.float32, device=device)
        logits = D(xb, pb).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def gan_local_motion_candidates(G, cond, norm, device, args, generator):
    cond_n = norm_X(cond[None, :], norm)
    bounds = {
        "step_min": norm["step_min"],
        "step_max": norm["step_max"],
        "speed_min": norm["speed_min"],
        "speed_max": norm["speed_max"],
    }
    with torch.no_grad():
        xb = torch.tensor(cond_n, dtype=torch.float32, device=device)
        z0 = torch.zeros(1, args.noise_dim, device=device)
        p0n = G(xb, z0).detach().cpu().numpy()
        random_count = max(0, args.candidates_per_expand - 1)
        if random_count > 0:
            xb_rep = xb.repeat(random_count, 1)
            z = args.planner_noise_scale * torch.randn(
                random_count, args.noise_dim, device=device, generator=generator
            )
            prn = G(xb_rep, z).detach().cpu().numpy()
            pn = np.vstack([p0n, prn])
        else:
            pn = p0n
    params = unnorm_params(pn, norm)
    X = np.repeat(cond[None, :], len(params), axis=0)
    states = params_to_state_np(X, params, bounds=bounds)
    return states.astype(np.float32)


def fallback_candidates(parent_state, goal_state, obstacles, args, rng, exploration_target=None):
    states = []
    parent_pos = parent_state[0:2]
    goal_pos = goal_state[0:2]
    to_goal = goal_pos - parent_pos
    dist_goal = float(np.linalg.norm(to_goal))
    step_len = min(args.max_step, max(args.min_step, dist_goal))
    effective_dt = float(getattr(args, "runtime_effective_dt_normalized", args.nominal_dt))
    nominal_speed = min(
        args.max_speed,
        max(args.min_speed, step_len / max(effective_dt, 1e-8)),
    )
    direct = state_from_direction(parent_state, to_goal, step_len, nominal_speed)
    if direct is not None:
        states.append(direct)
    if exploration_target is not None:
        sample_direction = np.asarray(exploration_target, dtype=np.float32) - parent_pos
        sample_state = state_from_direction(parent_state, sample_direction, step_len, nominal_speed)
        if sample_state is not None:
            states.append(sample_state)
    if dist_goal > 1e-8:
        goal_angle = math.atan2(to_goal[1], to_goal[0])
        # A deterministic fan supplies genuine detour choices when the direct
        # route is blocked. The wider angles are what let the tree leave a
        # local dead end instead of repeatedly touching the same obstacle.
        for offset_deg in (-105, -75, -50, -25, 25, 50, 75, 105):
            angle = goal_angle + math.radians(offset_deg)
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            s = state_from_direction(parent_state, direction, step_len, nominal_speed)
            if s is not None:
                states.append(s)
    nearby = sorted(
        obstacles,
        key=lambda ob: float(np.linalg.norm(parent_pos - np.asarray(ob[:2])) - ob[2]),
    )[:6]
    for ox, oy, rr in nearby:
        center = np.array([ox, oy], dtype=np.float32)
        obs_vec = parent_pos - center
        n = float(np.linalg.norm(obs_vec))
        if n < 1e-8:
            continue
        radial = obs_vec / n
        tangent1 = np.array([-radial[1], radial[0]], dtype=np.float32)
        tangent2 = np.array([radial[1], -radial[0]], dtype=np.float32)
        for t in [tangent1, tangent2]:
            blended = 0.35 * to_goal + 0.85 * t
            s = state_from_direction(parent_state, blended, step_len, nominal_speed)
            if s is not None:
                states.append(s)
    for _ in range(args.random_fallback_candidates):
        angle = float(rng.uniform(-math.pi, math.pi))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        random_step = float(rng.uniform(max(args.min_step, 0.55 * step_len), step_len))
        s = state_from_direction(parent_state, direction, random_step, nominal_speed)
        if s is not None:
            states.append(s)
    if not states:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(states, dtype=np.float32)


def sample_rrt_target(goal, obstacles, rng, args):
    """Sample the target before selecting a parent, exactly as basic RRT does."""
    if rng.random() < args.goal_sample_rate:
        return np.asarray(goal[0:2], dtype=np.float32), True
    for _ in range(args.sample_rejection_limit):
        sample = rng.uniform(args.workspace_min, args.workspace_max, size=2).astype(np.float32)
        if not point_is_invalid(
            sample, obstacles, args.collision_margin,
            args.workspace_min, args.workspace_max,
        ):
            return sample, False
    # Deterministic finite fallback: scan a shifted grid instead of returning
    # an unchecked point. This cannot hang and cannot silently violate the
    # basic-RRT requirement that samples lie in free space.
    grid_side = int(max(3, math.ceil(math.sqrt(args.sample_rejection_limit))))
    offset = int(rng.integers(0, grid_side * grid_side))
    for flat in range(grid_side * grid_side):
        cell = (flat + offset) % (grid_side * grid_side)
        ix, iy = divmod(cell, grid_side)
        sample = np.array(
            [
                args.workspace_min + (ix + 0.5) * (args.workspace_max - args.workspace_min) / grid_side,
                args.workspace_min + (iy + 0.5) * (args.workspace_max - args.workspace_min) / grid_side,
            ],
            dtype=np.float32,
        )
        if not point_is_invalid(
            sample, obstacles, args.collision_margin,
            args.workspace_min, args.workspace_max,
        ):
            return sample, False
    return None, False


def choose_parent(nodes, sample_xy, args):
    """Return the tree node nearest to the already sampled RRT target."""
    sample_xy = np.asarray(sample_xy, dtype=np.float32)
    return min(
        range(len(nodes)),
        key=lambda i: (
            float(np.linalg.norm(nodes[i].state[0:2] - sample_xy))
            + args.nearest_velocity_weight * float(np.linalg.norm(nodes[i].state[2:4]))
        ),
    )


def node_key(state, args):
    pos_res = max(float(args.duplicate_position_resolution), 1e-6)
    vel_res = max(float(args.duplicate_velocity_resolution), 1e-6)
    return (
        int(round(float(state[0]) / pos_res)),
        int(round(float(state[1]) / pos_res)),
        int(round(float(state[2]) / vel_res)),
        int(round(float(state[3]) / vel_res)),
    )


def point_is_invalid(pos, obstacles, margin, workspace_min=0.0, workspace_max=1.0):
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (2,) or not np.isfinite(pos).all():
        return True
    if (
        pos[0] < workspace_min + margin or pos[0] > workspace_max - margin
        or pos[1] < workspace_min + margin or pos[1] > workspace_max - margin
    ):
        return True
    return any(float(np.linalg.norm(pos - np.asarray([ox, oy]))) <= rr + margin for ox, oy, rr in obstacles)


def order_child_candidates(scored, rng, args):
    """Rank valid local steering candidates; basic RRT adds only the best one."""
    del rng, args
    return sorted(scored, key=lambda item: item[0], reverse=True)


def reconstruct_path(nodes, goal_idx):
    idx = goal_idx
    rev = []
    while idx >= 0:
        rev.append(nodes[idx].state.copy())
        idx = nodes[idx].parent
    rev.reverse()
    return np.asarray(rev, dtype=np.float32)


def score_local_candidates(
    raw_states, source, D, cond, parent, sample_xy, obstacles, norm,
    device, args, visited, use_discriminator,
):
    """Steer, safety-filter, and score one source of local proposals."""
    steered = []
    nonfinite_rejections = 0
    for raw in raw_states:
        if np.asarray(raw).shape != (4,) or not np.isfinite(raw).all():
            nonfinite_rejections += 1
            continue
        candidate = steer(parent.state, raw, args)
        if candidate is not None:
            steered.append(candidate)
    if not steered:
        return [], nonfinite_rejections, 0, 0, []
    steered = np.asarray(steered, dtype=np.float32)
    if use_discriminator:
        probs = discriminator_prob(D, cond, steered, norm, device)
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        probs = np.zeros(len(steered), dtype=np.float32)
    dist_now = float(np.linalg.norm(parent.state[0:2] - sample_xy))
    clearance_now = min_clearance(parent.state[0:2], obstacles)
    scored = []
    collision_boundary_rejections = 0
    duplicate_rejections = 0
    rejected_edges = []
    for i, cand in enumerate(steered):
        pos = cand[0:2]
        step_len = float(np.linalg.norm(pos - parent.state[0:2]))
        new_dist = float(np.linalg.norm(pos - sample_xy))
        progress = dist_now - new_dist
        turn = angle_between(parent.state[2:4], cand[2:4])
        invalid_edge = edge_is_invalid(
            parent.state[0:2], pos, obstacles, args.collision_margin,
            args.workspace_min, args.workspace_max,
        )
        if invalid_edge:
            collision_boundary_rejections += 1
            rejected_edges.append((parent.state.copy(), cand.copy()))
            continue
        clearance = min_clearance(pos, obstacles)
        if progress < args.min_progress:
            if clearance <= clearance_now + args.detour_clearance_gain:
                continue
        if node_key(cand, args) in visited:
            duplicate_rejections += 1
            continue
        heading_align = 0.0
        to_sample = sample_xy - parent.state[0:2]
        if np.linalg.norm(to_sample) > 1e-8 and np.linalg.norm(cand[2:4]) > 1e-8:
            heading_align = float(np.dot(
                to_sample / np.linalg.norm(to_sample),
                cand[2:4] / np.linalg.norm(cand[2:4]),
            ))
        score = (
            args.sample_progress_weight * progress
            + 0.30 * float(probs[i])
            + 0.25 * max(-0.1, min(clearance, 0.3))
            + 0.20 * heading_align
            - 0.20 * step_len
            - 0.28 * turn
            - 0.015 * parent.cost
        )
        scored.append((score, cand, step_len, source))
    return (
        scored, nonfinite_rejections, collision_boundary_rejections,
        duplicate_rejections, rejected_edges,
    )


def gan_rrt_plan(
    G, D, base_cond, goal_state, norm, device, args, seed, use_gan=True,
):
    if use_gan:
        G.eval()
        D.eval()
    # Each attempt receives its own deterministic RRT-sampling stream. Learned
    # and classical paired runs therefore get the same samples for attempt k,
    # even if an earlier attempt terminated at a different iteration.
    start = np.asarray(base_cond[0:4], dtype=np.float32)
    goal = np.asarray(goal_state, dtype=np.float32)
    if goal.shape != (4,):
        raise ValueError(f"goal_state must have shape (4,), got {goal.shape}")
    obstacles = parse_obstacles(base_cond)
    overall_start_time = time.perf_counter()
    invalid_reason = None
    if point_is_invalid(
        start[0:2], obstacles, args.collision_margin,
        args.workspace_min, args.workspace_max,
    ):
        invalid_reason = "invalid_start"
    elif point_is_invalid(
        goal[0:2], obstacles, args.collision_margin,
        args.workspace_min, args.workspace_max,
    ):
        invalid_reason = "invalid_goal"
    if invalid_reason is not None:
        node = Node(state=start.copy(), parent=-1, cost=0.0, depth=0)
        return {
            "planner_mode": "gan_with_conditional_safety_fallback" if use_gan else "classical_basic_rrt",
            "solved": False,
            "reason": invalid_reason,
            "attempt": 0,
            "nodes": [node],
            "edges": [],
            "rejected_edges": [],
            "path": np.asarray([start], dtype=np.float32),
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
            "collision_margin": float(args.collision_margin),
            "goal_radius": float(args.goal_radius),
            "generated_total": 0,
            "accepted_total": 0,
            "sampled_target_count": 0,
            "goal_sample_count": 0,
            "nearest_parent_selection_count": 0,
            "gan_condition_count": 0,
            "iterations_with_usable_gan_candidate": 0,
            "gan_accepted_total": 0,
            "fallback_accepted_total": 0,
            "nonfinite_candidate_rejections": 0,
            "sample_condition_mismatch_count": 0,
            "max_children_added_in_any_iteration": 0,
            "goal_connector_nodes": 0,
            "accepted_edge_collision_violations": 0,
            "free_sample_failures": 0,
            "runtime_contract_passed": True,
            "duplicate_rejections": 0,
            "collision_boundary_rejections": 0,
            "iterations": 0,
            "attempts_executed": 0,
            "total_iterations_across_attempts": 0,
            "total_sampled_targets_across_attempts": 0,
            "total_generated_candidates_across_attempts": 0,
            "total_accepted_nodes_across_attempts": 0,
            "best_distance_to_goal": float(np.linalg.norm(start[0:2] - goal[0:2])),
            "path_cost": 0.0,
            "runtime_seconds": float(time.perf_counter() - overall_start_time),
        }
    initial_distance = float(np.linalg.norm(start[0:2] - goal[0:2]))
    if initial_distance <= args.goal_radius:
        # This is a valid but trivial scenario: the root already lies in the
        # goal region. Do not append an unsampled synthetic goal node, because
        # even excluded benchmark cases must obey the one-sample/one-child
        # basic-RRT accounting contract.
        nodes = [Node(state=start.copy(), parent=-1, cost=0.0, depth=0)]
        edges = []
        return {
            "planner_mode": "gan_with_conditional_safety_fallback" if use_gan else "classical_basic_rrt",
            "solved": True,
            "reason": "start_within_goal_radius",
            "attempt": 0,
            "nodes": nodes,
            "edges": edges,
            "rejected_edges": [],
            "path": np.asarray([node.state for node in nodes], dtype=np.float32),
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
            "collision_margin": float(args.collision_margin),
            "goal_radius": float(args.goal_radius),
            "generated_total": 0,
            "accepted_total": 0,
            "sampled_target_count": 0,
            "goal_sample_count": 0,
            "nearest_parent_selection_count": 0,
            "gan_condition_count": 0,
            "iterations_with_usable_gan_candidate": 0,
            "gan_accepted_total": 0,
            "fallback_accepted_total": 0,
            "nonfinite_candidate_rejections": 0,
            "sample_condition_mismatch_count": 0,
            "max_children_added_in_any_iteration": 0,
            "goal_connector_nodes": 0,
            "accepted_edge_collision_violations": 0,
            "free_sample_failures": 0,
            "runtime_contract_passed": True,
            "duplicate_rejections": 0,
            "collision_boundary_rejections": 0,
            "iterations": 0,
            "attempts_executed": 0,
            "total_iterations_across_attempts": 0,
            "total_sampled_targets_across_attempts": 0,
            "total_generated_candidates_across_attempts": 0,
            "total_accepted_nodes_across_attempts": 0,
            "best_distance_to_goal": initial_distance,
            "path_cost": 0.0,
            "runtime_seconds": float(time.perf_counter() - overall_start_time),
        }
    best_result = None
    attempt_effort = []
    for attempt in range(args.attempts_per_case):
        sample_rng = np.random.default_rng(seed + 104729 * (attempt + 1))
        fallback_rng = np.random.default_rng(
            seed + 32452843 + 130363 * (attempt + 1)
        )
        nodes = [Node(state=start.copy(), parent=-1, cost=0.0, depth=0)]
        visited = {node_key(start, args)}
        edges = []
        rejected_edges = []
        generated_total = 0
        accepted_total = 0
        duplicate_rejections = 0
        collision_boundary_rejections = 0
        sampled_target_count = 0
        goal_sample_count = 0
        nearest_parent_selection_count = 0
        gan_condition_count = 0
        iterations_with_usable_gan_candidate = 0
        gan_accepted_total = 0
        fallback_accepted_total = 0
        nonfinite_candidate_rejections = 0
        sample_condition_mismatch_count = 0
        max_children_added_in_any_iteration = 0
        accepted_edge_collision_violations = 0
        free_sample_failures = 0
        goal_idx = None
        reason = "max_iterations"
        it = -1
        torch_generator = (
            make_torch_generator(device, seed + 1000003 * (attempt + 1))
            if use_gan else None
        )
        for it in range(args.max_iterations):
            # Correct RRT order: sample first, choose the nearest existing
            # parent second, then condition the GAN on that exact pair.
            sample_xy, sampled_goal = sample_rrt_target(
                goal, obstacles, sample_rng, args
            )
            if sample_xy is None:
                free_sample_failures += 1
                reason = "no_free_sample_available"
                continue
            sampled_target_count += 1
            goal_sample_count += int(sampled_goal)
            parent_idx = choose_parent(nodes, sample_xy, args)
            nearest_parent_selection_count += 1
            parent = nodes[parent_idx]
            parent.expansions += 1
            cond = condition_from_node(base_cond, parent.state, sample_xy)
            candidate_pool = []
            if use_gan:
                gan_condition_count += 1
                if not np.array_equal(
                    cond[4:6], np.asarray(sample_xy, dtype=np.float32)
                ):
                    sample_condition_mismatch_count += 1
                gan_states = gan_local_motion_candidates(
                    G, cond, norm, device, args, torch_generator
                )
                generated_total += len(gan_states)
                (
                    candidate_pool, nonfinite_count, collision_count,
                    duplicate_count, rejected,
                ) = score_local_candidates(
                    gan_states, "gan", D, cond, parent, sample_xy,
                    obstacles, norm, device, args, visited,
                    use_discriminator=True,
                )
                nonfinite_candidate_rejections += nonfinite_count
                collision_boundary_rejections += collision_count
                duplicate_rejections += duplicate_count
                remaining_rejected = max(
                    0, args.max_rejected_edges_stored - len(rejected_edges)
                )
                rejected_edges.extend(rejected[:remaining_rejected])
                if candidate_pool:
                    iterations_with_usable_gan_candidate += 1

            # Generate fallback actions only when the learned pool is empty.
            # This both enforces GAN priority and avoids consuming random
            # numbers or computation for candidates that cannot be selected.
            if not candidate_pool:
                if use_gan:
                    local_target_state = np.zeros(4, dtype=np.float32)
                    local_target_state[0:2] = sample_xy
                    fallback_states = fallback_candidates(
                        parent.state, local_target_state, obstacles, args, fallback_rng,
                        exploration_target=None,
                    )
                else:
                    sample_direction = sample_xy - parent.state[0:2]
                    sample_distance = float(np.linalg.norm(sample_direction))
                    if sample_distance >= args.min_step:
                        step_len = min(args.max_step, sample_distance)
                        effective_dt = float(getattr(
                            args, "runtime_effective_dt_normalized", args.nominal_dt
                        ))
                        speed = float(np.clip(
                            step_len / max(effective_dt, 1e-8),
                            args.min_speed, args.max_speed,
                        ))
                        classical_state = state_from_direction(
                            parent.state, sample_direction, step_len, speed
                        )
                        fallback_states = np.asarray(
                            [classical_state], dtype=np.float32
                        )
                    else:
                        fallback_states = np.zeros((0, 4), dtype=np.float32)
                generated_total += len(fallback_states)
                (
                    candidate_pool, nonfinite_count, collision_count,
                    duplicate_count, rejected,
                ) = score_local_candidates(
                    fallback_states, "fallback", D, cond, parent, sample_xy,
                    obstacles, norm, device, args, visited,
                    use_discriminator=bool(use_gan),
                )
                nonfinite_candidate_rejections += nonfinite_count
                collision_boundary_rejections += collision_count
                duplicate_rejections += duplicate_count
                remaining_rejected = max(
                    0, args.max_rejected_edges_stored - len(rejected_edges)
                )
                rejected_edges.extend(rejected[:remaining_rejected])
            ordered_candidates = order_child_candidates(
                candidate_pool, fallback_rng, args
            )
            children_added = 0
            for score, cand, step_len, source in ordered_candidates:
                if children_added >= args.children_per_expand:
                    break
                if len(nodes) >= args.max_nodes:
                    reason = "max_nodes"
                    break
                key = node_key(cand, args)
                if key in visited:
                    # Defensive re-check: the candidate pool was already
                    # duplicate-free, but the insertion boundary owns safety.
                    duplicate_rejections += 1
                    continue
                # Re-check the exact accepted edge at the insertion boundary.
                # Package validity depends on this counter remaining zero.
                if edge_is_invalid(
                    parent.state[0:2], cand[0:2], obstacles,
                    args.collision_margin, args.workspace_min, args.workspace_max,
                ):
                    accepted_edge_collision_violations += 1
                    continue
                visited.add(key)
                nodes.append(Node(
                    state=cand.copy(), parent=parent_idx,
                    cost=parent.cost + step_len, depth=parent.depth + 1,
                    source=source,
                ))
                child_idx = len(nodes) - 1
                edges.append((parent_idx, child_idx))
                accepted_total += 1
                if source == "gan":
                    gan_accepted_total += 1
                else:
                    fallback_accepted_total += 1
                children_added += 1
                max_children_added_in_any_iteration = max(
                    max_children_added_in_any_iteration, children_added
                )
                if np.linalg.norm(cand[0:2] - goal[0:2]) <= args.goal_radius:
                    # Standard basic RRT terminates in a goal region. No hidden
                    # RRT-Connect chain or second node is appended here.
                    goal_idx = child_idx
                    reason = "goal_region_reached"
                    break
            if goal_idx is not None:
                break
            if len(nodes) >= args.max_nodes:
                reason = "max_nodes"
                break
        solved = goal_idx is not None
        if goal_idx is not None:
            best_idx = int(goal_idx)
            path = reconstruct_path(nodes, best_idx)
        else:
            best_idx = int(np.argmin([np.linalg.norm(n.state[0:2] - goal[0:2]) for n in nodes]))
            path = reconstruct_path(nodes, best_idx)
        runtime_contract_passed = bool(
            sampled_target_count == nearest_parent_selection_count
            and (
                sampled_target_count == gan_condition_count
                if use_gan else gan_condition_count == 0
            )
            and sample_condition_mismatch_count == 0
            and max_children_added_in_any_iteration <= 1
            and accepted_total <= sampled_target_count
            and accepted_total == gan_accepted_total + fallback_accepted_total
            and accepted_edge_collision_violations == 0
        )
        if not runtime_contract_passed:
            raise RuntimeError(
                "Basic-RRT runtime contract failed: "
                f"mode={'gan' if use_gan else 'classical'}, "
                f"samples={sampled_target_count}, parents={nearest_parent_selection_count}, "
                f"gan_conditions={gan_condition_count}, mismatches={sample_condition_mismatch_count}, "
                f"max_children={max_children_added_in_any_iteration}, accepted={accepted_total}, "
                f"gan_accepted={gan_accepted_total}, fallback_accepted={fallback_accepted_total}, "
                f"collision_violations={accepted_edge_collision_violations}"
            )
        result = {
            "planner_mode": "gan_with_conditional_safety_fallback" if use_gan else "classical_basic_rrt",
            "solved": bool(solved),
            "reason": reason,
            "attempt": int(attempt + 1),
            "nodes": nodes,
            "edges": edges,
            "rejected_edges": rejected_edges,
            "path": path,
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
            "collision_margin": float(args.collision_margin),
            "goal_radius": float(args.goal_radius),
            "generated_total": int(generated_total),
            "accepted_total": int(accepted_total),
            "sampled_target_count": int(sampled_target_count),
            "goal_sample_count": int(goal_sample_count),
            "nearest_parent_selection_count": int(nearest_parent_selection_count),
            "gan_condition_count": int(gan_condition_count),
            "iterations_with_usable_gan_candidate": int(iterations_with_usable_gan_candidate),
            "gan_accepted_total": int(gan_accepted_total),
            "fallback_accepted_total": int(fallback_accepted_total),
            "nonfinite_candidate_rejections": int(nonfinite_candidate_rejections),
            "sample_condition_mismatch_count": int(sample_condition_mismatch_count),
            "max_children_added_in_any_iteration": int(max_children_added_in_any_iteration),
            "goal_connector_nodes": 0,
            "accepted_edge_collision_violations": int(accepted_edge_collision_violations),
            "free_sample_failures": int(free_sample_failures),
            "runtime_contract_passed": runtime_contract_passed,
            "duplicate_rejections": int(duplicate_rejections),
            "collision_boundary_rejections": int(collision_boundary_rejections),
            "iterations": int(it + 1),
            "best_distance_to_goal": float(np.linalg.norm(nodes[best_idx].state[0:2] - goal[0:2])),
            "path_cost": float(nodes[best_idx].cost),
            "runtime_seconds": float(time.perf_counter() - overall_start_time),
        }
        attempt_effort.append({
            "iterations": int(result["iterations"]),
            "sampled_targets": int(result["sampled_target_count"]),
            "generated_candidates": int(result["generated_total"]),
            "accepted_nodes": int(result["accepted_total"]),
        })
        if best_result is None:
            best_result = result
        else:
            if result["solved"] and not best_result["solved"]:
                best_result = result
            elif result["solved"] == best_result["solved"]:
                if result["best_distance_to_goal"] < best_result["best_distance_to_goal"]:
                    best_result = result
                elif abs(result["best_distance_to_goal"] - best_result["best_distance_to_goal"]) < 1e-9 and result["path_cost"] < best_result["path_cost"]:
                    best_result = result
        if best_result["solved"] and attempt + 1 >= args.min_attempts_before_accept:
            break
    if best_result is None:
        raise RuntimeError("Planner executed no attempts despite validated arguments")
    best_result["attempts_executed"] = int(len(attempt_effort))
    best_result["total_iterations_across_attempts"] = int(
        sum(item["iterations"] for item in attempt_effort)
    )
    best_result["total_sampled_targets_across_attempts"] = int(
        sum(item["sampled_targets"] for item in attempt_effort)
    )
    best_result["total_generated_candidates_across_attempts"] = int(
        sum(item["generated_candidates"] for item in attempt_effort)
    )
    best_result["total_accepted_nodes_across_attempts"] = int(
        sum(item["accepted_nodes"] for item in attempt_effort)
    )
    best_result["runtime_seconds"] = float(
        time.perf_counter() - overall_start_time
    )
    return best_result


def plot_gan_rrt(result, out_path, title):
    out_path = Path(out_path)
    mkdir(out_path.parent)
    fig, ax = plt.subplots(figsize=(7.8, 7.5), facecolor="white")
    ax.set_facecolor("white")
    safety_margin = float(result.get("collision_margin", 0.0))
    safe_min = safety_margin
    safe_span = 1.0 - 2.0 * safety_margin
    if safe_span > 0.0:
        ax.add_patch(Rectangle(
            (safe_min, safe_min), safe_span, safe_span, fill=False,
            linewidth=1.1, linestyle=":", edgecolor="#6D6E71",
            label="Workspace configuration-space boundary",
        ))
    for obstacle_index, (ox, oy, rr) in enumerate(result["obstacles"]):
        ax.add_patch(Circle(
            (ox, oy), rr, alpha=0.72, facecolor="#B7BBC0",
            edgecolor="#54585C", linewidth=0.8,
        ))
        ax.add_patch(Circle(
            (ox, oy), rr + safety_margin, fill=False, linewidth=1.2,
            linestyle="--", edgecolor="#54585C",
            label="Configuration-space safety boundary" if obstacle_index == 0 else None,
        ))
    nodes = result["nodes"]
    labeled_sources = set()
    for parent_idx, child_idx in result["edges"]:
        a = nodes[parent_idx].state
        b = nodes[child_idx].state
        source = getattr(nodes[child_idx], "source", "gan")
        edge_color = "#6CACE4" if source == "gan" else "#8A8D8F"
        edge_label = (
            "GAN-steered tree branches" if source == "gan"
            else "Safety-fallback tree branches"
        )
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], linewidth=0.75, alpha=0.52,
            color=edge_color,
            label=edge_label if source not in labeled_sources else None,
        )
        labeled_sources.add(source)
    for rejected_number, (a, b) in enumerate(result["rejected_edges"][:250]):
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], linewidth=0.5, alpha=0.20,
            color="#B76E79", label="Rejected proposal" if rejected_number == 0 else None,
        )
    p = result["path"]
    if len(p) >= 2:
        path_label = "Final solved path" if result["solved"] else "Best partial path"
        ax.plot(
            p[:, 0], p[:, 1], linewidth=3.2, color="#0021A5",
            label=path_label, zorder=8,
        )
    ax.scatter(
        [result["start"][0]], [result["start"][1]], marker="o", s=85,
        color="#252525", label="Start", zorder=9,
    )
    ax.scatter(
        [result["goal"][0]], [result["goal"][1]], marker="*", s=150,
        color="#00529B", edgecolor="white", linewidth=0.7,
        label="Goal", zorder=10,
    )
    ax.add_patch(Circle(
        (result["goal"][0], result["goal"][1]),
        float(result.get("goal_radius", 0.0)),
        fill=False, linewidth=1.4, linestyle="--", edgecolor="#00529B",
        label="Goal region",
    ))
    ax.text(result["start"][0] - 0.02, result["start"][1] - 0.02, "Start", fontsize=9.5, ha="right", va="top")
    ax.text(result["goal"][0] + 0.02, result["goal"][1] + 0.02, "Goal", fontsize=9.5, ha="left", va="bottom")
    status_word = "SOLVED" if result["solved"] else "FAILED"
    subtitle_1 = (
        f"{status_word} ({result['reason']}) | selected attempt "
        f"{result['attempt']}/{result['attempts_executed']} | "
        f"nodes={len(nodes)} | selected-attempt iterations={result['iterations']}"
    )
    subtitle_2 = (
        f"accepted GAN/fallback={result['gan_accepted_total']}/{result['fallback_accepted_total']} | "
        f"collision/boundary rejected={result['collision_boundary_rejections']} | "
        f"samples={result['sampled_target_count']} (goal={result['goal_sample_count']}) | "
        f"total runtime={result['runtime_seconds']:.3f}s"
    )
    ax.set_title(title, fontsize=14, weight="semibold", color="#252525", pad=10)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D8DCE1", linewidth=0.65, alpha=0.65)
    ax.set_xlabel("Normalized x position")
    ax.set_ylabel("Normalized y position")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(
        0.5, -0.105, f"{subtitle_1}\n{subtitle_2}",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.3,
        color="#3F4144",
    )
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(
        unique.values(), unique.keys(), loc="upper center",
        bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def difficulty_score(cond):
    start = np.asarray(cond[0:2], dtype=float)
    goal = np.asarray(cond[4:6], dtype=float)
    obs = parse_obstacles(cond)
    dist = float(np.linalg.norm(goal - start))
    if not obs:
        return dist
    clearances = [point_segment_distance((ox, oy), start, goal) - rr for ox, oy, rr in obs]
    closest = min(clearances)
    return dist + max(0.0, 0.25 - closest) * 4.0 + 0.06 * len(obs)


def classify_scenario(cond, goal_state, args):
    """Classify benchmark eligibility before a planner is credited/blamed."""
    start = np.asarray(cond[0:2], dtype=np.float32)
    goal = np.asarray(goal_state[0:2], dtype=np.float32)
    obstacles = parse_obstacles(cond)
    if point_is_invalid(
        start, obstacles, args.collision_margin,
        args.workspace_min, args.workspace_max,
    ):
        return "invalid_start"
    if point_is_invalid(
        goal, obstacles, args.collision_margin,
        args.workspace_min, args.workspace_max,
    ):
        return "invalid_goal"
    if float(np.linalg.norm(start - goal)) <= args.goal_radius:
        return "trivial_start_within_goal_radius"
    return "valid_nontrivial"


def benchmark_eligibility_for_split(X, goals, args):
    counts = {
        "invalid_start": 0,
        "invalid_goal": 0,
        "trivial_start_within_goal_radius": 0,
        "valid_nontrivial": 0,
    }
    eligible = []
    for idx in range(len(X)):
        reason = classify_scenario(X[idx], goals[idx], args)
        counts[reason] += 1
        if reason == "valid_nontrivial":
            eligible.append(idx)
    total = int(len(X))
    report = {
        "rows": total,
        **{key: int(value) for key, value in counts.items()},
        **{
            f"{key}_percent": float(100.0 * value / max(1, total))
            for key, value in counts.items()
        },
        "benchmark_policy": (
            "Only valid_nontrivial scenarios enter solve-rate and difficulty benchmarks; "
            "invalid and zero-iteration trivial rows are reported separately."
        ),
    }
    return eligible, report


def benchmark_eligibility_diagnostics(data, args):
    out = {}
    for split, xkey, goal_key in [
        ("training", "train_X", "training_goal_state"),
        ("validation", "validation_X", "validation_goal_state"),
        ("testing", "testing_X", "testing_goal_state"),
    ]:
        _, out[split] = benchmark_eligibility_for_split(
            data[xkey], data[goal_key], args
        )
    return out


def selected_benchmark_cases(X, goals, cases_per_difficulty, seed, args):
    """Uniformly sample exact easy/medium/hard counts from valid cases only."""
    eligible, eligibility = benchmark_eligibility_for_split(X, goals, args)
    scenario_conditions = np.asarray(X, dtype=np.float32).copy()
    scenario_conditions[:, 4:6] = np.asarray(goals, dtype=np.float32)[:, 0:2]
    scored = sorted(
        [(difficulty_score(scenario_conditions[idx]), int(idx)) for idx in eligible],
        key=lambda item: item[0],
    )
    bucket_positions = np.array_split(np.arange(len(scored), dtype=int), 3)
    buckets = {
        name: [scored[int(position)] for position in positions]
        for name, positions in zip(
            ("easy", "medium", "hard"), bucket_positions
        )
    }
    rng = np.random.default_rng(seed)
    selected = []
    for name in ("easy", "medium", "hard"):
        rows = buckets[name]
        if len(rows) < cases_per_difficulty:
            raise RuntimeError(
                f"{name}: requested {cases_per_difficulty} valid nontrivial benchmark cases, "
                f"but only {len(rows)} are available. Eligibility: {eligibility}"
            )
        chosen_positions = rng.choice(
            len(rows), size=cases_per_difficulty, replace=False
        )
        for position in chosen_positions:
            score, idx = rows[int(position)]
            selected.append((int(idx), name, float(score)))
    return selected, eligibility


def wilson_interval_percent(successes, total, z=1.959963984540054):
    if total <= 0:
        return None, None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return float(100.0 * (center - half)), float(100.0 * (center + half))


def clustered_bootstrap_solve_interval_percent(
    rows, cluster_key="environment_id", seed=0, replicates=2000
):
    """Percentile interval from resampling whole environments.

    Multiple transition rows from one environment are correlated. Resampling
    environment clusters preserves that dependence instead of pretending every
    rollout row is an independent Bernoulli trial.
    """
    clusters = {}
    for row in rows:
        clusters.setdefault(int(row[cluster_key]), []).append(bool(row["solved"]))
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2 or replicates < 1:
        return None, None
    rng = np.random.default_rng(int(seed))
    rates = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        chosen = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        solved = 0
        count = 0
        for cluster_position in chosen:
            values = clusters[cluster_ids[int(cluster_position)]]
            solved += int(sum(values))
            count += len(values)
        rates[replicate] = 100.0 * solved / max(1, count)
    low, high = np.percentile(rates, [2.5, 97.5])
    return float(low), float(high)


def summarize_rollout_rows(rows, split, bootstrap_seed=0):
    def one(group_rows, salt):
        solved_rows = [r for r in group_rows if r["solved"]]
        ci_low, ci_high = wilson_interval_percent(len(solved_rows), len(group_rows))
        cluster_low, cluster_high = clustered_bootstrap_solve_interval_percent(
            group_rows,
            cluster_key="environment_id",
            seed=int(bootstrap_seed) + int(salt),
        )
        return {
            "cases": int(len(group_rows)),
            "solved": int(len(solved_rows)),
            "failed": int(len(group_rows) - len(solved_rows)),
            "solve_rate_percent": float(100.0 * len(solved_rows) / max(1, len(group_rows))),
            "solve_rate_wilson_95_low_percent": ci_low,
            "solve_rate_wilson_95_high_percent": ci_high,
            "solve_rate_environment_cluster_bootstrap_95_low_percent": cluster_low,
            "solve_rate_environment_cluster_bootstrap_95_high_percent": cluster_high,
            "environment_cluster_bootstrap_replicates": 2000,
            "unique_environment_clusters": int(len({
                int(r["environment_id"]) for r in group_rows
            })),
            "unique_scenario_clusters": int(len({
                int(r["scenario_id"]) for r in group_rows
            })),
            "mean_solved_path_cost": float(np.mean([r["path_cost"] for r in solved_rows])) if solved_rows else None,
            "median_solved_path_cost": float(np.median([r["path_cost"] for r in solved_rows])) if solved_rows else None,
            "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in group_rows])) if group_rows else None,
            "mean_nodes": float(np.mean([r["nodes"] for r in group_rows])) if group_rows else None,
            "mean_iterations": float(np.mean([r["iterations"] for r in group_rows])) if group_rows else None,
            "mean_total_iterations_across_attempts": float(np.mean(
                [r["total_iterations_across_attempts"] for r in group_rows]
            )) if group_rows else None,
            "mean_best_distance_to_goal": float(np.mean([r["best_distance_to_goal"] for r in group_rows])) if group_rows else None,
            "gan_accepted_nodes": int(
                sum(int(r["gan_accepted_total"]) for r in group_rows)
            ),
            "fallback_accepted_nodes": int(
                sum(int(r["fallback_accepted_total"]) for r in group_rows)
            ),
            "gan_share_of_accepted_nodes_percent": float(
                100.0
                * sum(int(r["gan_accepted_total"]) for r in group_rows)
                / max(1, sum(int(r["accepted_total"]) for r in group_rows))
            ) if group_rows else None,
            "runtime_contract_pass_percent": float(
                100.0 * np.mean([bool(r["runtime_contract_passed"]) for r in group_rows])
            ) if group_rows else None,
            "accepted_edge_collision_violations": int(
                sum(int(r["accepted_edge_collision_violations"]) for r in group_rows)
            ),
        }

    summary = {"split": split, **one(rows, 0), "by_difficulty": {}}
    for bucket_index, bucket in enumerate(("easy", "medium", "hard"), start=1):
        summary["by_difficulty"][bucket] = one(
            [r for r in rows if r["difficulty_bucket"] == bucket],
            1009 * bucket_index,
        )
    return summary


def generate_split_visuals(
    split, X, goals, environment_ids, scenario_ids,
    G, D, norm, device, args, section, seed,
):
    split_benchmark_started = time.perf_counter()
    section = Path(section)
    pics_root = section / "02_V7_STRICT_RRT_SAMPLE_GAN_PICTURES"
    data_root = section / "03_ROLLOUT_DATA"
    mkdir(pics_root)
    mkdir(data_root)
    selected, eligibility = selected_benchmark_cases(
        X, goals, args.benchmark_cases_per_difficulty, seed, args
    )
    render_budget = min(int(args.pictures_per_split), len(selected))
    render_keys = set()
    base = render_budget // 3
    remainder = render_budget % 3
    for bucket_index, bucket in enumerate(("easy", "medium", "hard")):
        group = [item for item in selected if item[1] == bucket]
        count = min(len(group), base + int(bucket_index < remainder))
        if count > 0:
            positions = np.linspace(0, len(group) - 1, num=count, dtype=int)
            render_keys.update((group[int(pos)][0], bucket) for pos in positions)
    rows = []
    pictures_written = 0
    for k, (idx, bucket, difficulty_value) in enumerate(selected):
        bucket_dir = pics_root / f"{bucket.upper()}_CASES"
        mkdir(bucket_dir)
        result = gan_rrt_plan(
            G, D, X[idx].astype(np.float32), goals[idx].astype(np.float32),
            norm, device, args, seed + idx,
        )
        picture_file = ""
        if (idx, bucket) in render_keys:
            pic_name = f"{split}_{bucket}_row_{idx:05d}_{'SOLVED' if result['solved'] else 'FAILED'}.png"
            pic_path = bucket_dir / pic_name
            plot_gan_rrt(
                result, pic_path,
                title=f"V7 GAN-RRT — {split.title()} / {bucket.title()} / row {idx}",
            )
            picture_file = str(pic_path.relative_to(section))
            pictures_written += 1
        rows.append({
            "split": split,
            "row_index": int(idx),
            "environment_id": int(environment_ids[idx]),
            "scenario_id": int(scenario_ids[idx]),
            "difficulty_bucket": bucket,
            "difficulty_score": difficulty_value,
            "scenario_eligibility": "valid_nontrivial",
            "picture_file": picture_file,
            "solved": bool(result["solved"]),
            "reason": result["reason"],
            "attempt": result["attempt"],
            "attempts_executed": result["attempts_executed"],
            "nodes": len(result["nodes"]),
            "generated_total": result["generated_total"],
            "accepted_total": result["accepted_total"],
            "sampled_target_count": result["sampled_target_count"],
            "goal_sample_count": result["goal_sample_count"],
            "nearest_parent_selection_count": result["nearest_parent_selection_count"],
            "gan_condition_count": result["gan_condition_count"],
            "iterations_with_usable_gan_candidate": result["iterations_with_usable_gan_candidate"],
            "gan_accepted_total": result["gan_accepted_total"],
            "fallback_accepted_total": result["fallback_accepted_total"],
            "nonfinite_candidate_rejections": result["nonfinite_candidate_rejections"],
            "gan_share_of_accepted_nodes_percent": float(
                100.0 * result["gan_accepted_total"] / max(1, result["accepted_total"])
            ),
            "sample_condition_mismatch_count": result["sample_condition_mismatch_count"],
            "max_children_added_in_any_iteration": result["max_children_added_in_any_iteration"],
            "goal_connector_nodes": result["goal_connector_nodes"],
            "accepted_edge_collision_violations": result["accepted_edge_collision_violations"],
            "free_sample_failures": result["free_sample_failures"],
            "runtime_contract_passed": result["runtime_contract_passed"],
            "duplicate_rejections": result["duplicate_rejections"],
            "collision_boundary_rejections": result["collision_boundary_rejections"],
            "iterations": result["iterations"],
            "total_iterations_across_attempts": result["total_iterations_across_attempts"],
            "total_sampled_targets_across_attempts": result["total_sampled_targets_across_attempts"],
            "total_generated_candidates_across_attempts": result["total_generated_candidates_across_attempts"],
            "total_accepted_nodes_across_attempts": result["total_accepted_nodes_across_attempts"],
            "best_distance_to_goal": result["best_distance_to_goal"],
            "path_cost": result["path_cost"],
            "path_length": result["path_cost"],
            "runtime_seconds": result["runtime_seconds"],
        })
        if (k + 1) % 10 == 0 or k + 1 == len(selected):
            print(
                measured_progress(
                    f"{split.upper()} GAN-RRT BENCHMARK",
                    k + 1,
                    len(selected),
                    split_benchmark_started,
                )
                + f" | pictures={pictures_written}/{render_budget}",
                flush=True,
            )
    write_csv(data_root / f"{split}_v7_strict_rrt_sample_gan_rollouts.csv", rows)
    summary = summarize_rollout_rows(rows, split, bootstrap_seed=seed + 700001)
    summary["pictures_written"] = int(pictures_written)
    summary["benchmark_cases_per_difficulty"] = int(args.benchmark_cases_per_difficulty)
    unique_scenarios = len({
        physical_scenario_signature(X[idx], goals[idx])
        for idx, _, _ in selected
    })
    summary["unique_physical_scenario_signatures"] = int(unique_scenarios)
    summary["benchmark_rows_per_unique_physical_scenario"] = float(
        len(selected) / max(1, unique_scenarios)
    )
    summary["eligibility_diagnostics_for_full_split"] = eligibility
    summary["measured_benchmark_wall_seconds"] = float(
        time.perf_counter() - split_benchmark_started
    )
    save_json(data_root / f"{split}_rollout_summary.json", summary)
    difficulty_rows = []
    for bucket, values in summary["by_difficulty"].items():
        difficulty_rows.append({"split": split, "difficulty": bucket, **values})
    write_csv(data_root / f"{split}_rollout_summary_by_difficulty.csv", difficulty_rows)
    return summary, rows


def generate_classical_testing_baseline(
    X, goals, learned_rows, args, overall_dir, seed,
):
    """Run classical basic RRT on the exact held-out cases used by the GAN."""
    baseline_started = time.perf_counter()
    data_dir = Path(overall_dir) / "02_OVERALL_BENCHMARK_DATA"
    mkdir(data_dir)
    rows = []
    learned_by_index = {int(row["row_index"]): row for row in learned_rows}
    for case_number, learned in enumerate(learned_rows, start=1):
        idx = int(learned["row_index"])
        result = gan_rrt_plan(
            None, None, X[idx].astype(np.float32), goals[idx].astype(np.float32),
            None, None, args, seed + idx, use_gan=False,
        )
        if learned["solved"] and result["solved"]:
            paired_outcome = "both_solved"
        elif learned["solved"]:
            paired_outcome = "gan_only_solved"
        elif result["solved"]:
            paired_outcome = "classical_only_solved"
        else:
            paired_outcome = "neither_solved"
        rows.append({
            "split": "testing_classical_baseline",
            "row_index": idx,
            "environment_id": int(learned["environment_id"]),
            "scenario_id": int(learned["scenario_id"]),
            "difficulty_bucket": learned["difficulty_bucket"],
            "difficulty_score": learned["difficulty_score"],
            "scenario_eligibility": "valid_nontrivial",
            "solved": bool(result["solved"]),
            "reason": result["reason"],
            "paired_outcome": paired_outcome,
            "gan_solved_same_case": bool(learned["solved"]),
            "gan_path_cost_same_case": learned["path_cost"],
            "gan_runtime_seconds_same_case": learned["runtime_seconds"],
            "attempt": result["attempt"],
            "attempts_executed": result["attempts_executed"],
            "nodes": len(result["nodes"]),
            "generated_total": result["generated_total"],
            "accepted_total": result["accepted_total"],
            "sampled_target_count": result["sampled_target_count"],
            "goal_sample_count": result["goal_sample_count"],
            "nearest_parent_selection_count": result["nearest_parent_selection_count"],
            "gan_condition_count": result["gan_condition_count"],
            "gan_accepted_total": result["gan_accepted_total"],
            "fallback_accepted_total": result["fallback_accepted_total"],
            "nonfinite_candidate_rejections": result["nonfinite_candidate_rejections"],
            "sample_condition_mismatch_count": result["sample_condition_mismatch_count"],
            "max_children_added_in_any_iteration": result["max_children_added_in_any_iteration"],
            "goal_connector_nodes": result["goal_connector_nodes"],
            "accepted_edge_collision_violations": result["accepted_edge_collision_violations"],
            "free_sample_failures": result["free_sample_failures"],
            "runtime_contract_passed": result["runtime_contract_passed"],
            "duplicate_rejections": result["duplicate_rejections"],
            "collision_boundary_rejections": result["collision_boundary_rejections"],
            "iterations": result["iterations"],
            "total_iterations_across_attempts": result["total_iterations_across_attempts"],
            "total_sampled_targets_across_attempts": result["total_sampled_targets_across_attempts"],
            "total_generated_candidates_across_attempts": result["total_generated_candidates_across_attempts"],
            "total_accepted_nodes_across_attempts": result["total_accepted_nodes_across_attempts"],
            "best_distance_to_goal": result["best_distance_to_goal"],
            "path_cost": result["path_cost"],
            "path_length": result["path_cost"],
            "runtime_seconds": result["runtime_seconds"],
        })
        if case_number % 10 == 0 or case_number == len(learned_rows):
            print(
                measured_progress(
                    "TESTING CLASSICAL-RRT PAIRED BASELINE",
                    case_number,
                    len(learned_rows),
                    baseline_started,
                ),
                flush=True,
            )
    if set(learned_by_index) != {int(row["row_index"]) for row in rows}:
        raise RuntimeError("Classical baseline did not evaluate the exact GAN testing rows")
    summary = summarize_rollout_rows(
        rows, "testing_classical_baseline", bootstrap_seed=seed + 800003
    )
    summary["measured_benchmark_wall_seconds"] = float(
        time.perf_counter() - baseline_started
    )
    unique_scenarios = len({
        physical_scenario_signature(X[int(row["row_index"])], goals[int(row["row_index"])])
        for row in rows
    })
    summary["unique_physical_scenario_signatures"] = int(unique_scenarios)
    summary["benchmark_rows_per_unique_physical_scenario"] = float(
        len(rows) / max(1, unique_scenarios)
    )
    both = [row for row in rows if row["paired_outcome"] == "both_solved"]
    comparison = {
        "paired_testing_cases": int(len(rows)),
        "same_row_indices_and_seeds": True,
        "both_solved": int(len(both)),
        "gan_only_solved": int(sum(
            row["paired_outcome"] == "gan_only_solved" for row in rows
        )),
        "classical_only_solved": int(sum(
            row["paired_outcome"] == "classical_only_solved" for row in rows
        )),
        "neither_solved": int(sum(
            row["paired_outcome"] == "neither_solved" for row in rows
        )),
        "gan_solve_rate_percent": float(
            100.0 * sum(bool(row["gan_solved_same_case"]) for row in rows)
            / max(1, len(rows))
        ),
        "classical_solve_rate_percent": float(summary["solve_rate_percent"]),
        "gan_minus_classical_solve_rate_percentage_points": float(
            100.0
            * (
                sum(bool(row["gan_solved_same_case"]) for row in rows)
                - sum(bool(row["solved"]) for row in rows)
            )
            / max(1, len(rows))
        ),
        "mean_gan_path_cost_on_both_solved": float(np.mean([
            float(row["gan_path_cost_same_case"]) for row in both
        ])) if both else None,
        "mean_classical_path_cost_on_both_solved": float(np.mean([
            float(row["path_cost"]) for row in both
        ])) if both else None,
        "mean_gan_runtime_seconds_all_cases": float(np.mean([
            float(row["gan_runtime_seconds_same_case"]) for row in rows
        ])) if rows else None,
        "mean_classical_runtime_seconds_all_cases": float(np.mean([
            float(row["runtime_seconds"]) for row in rows
        ])) if rows else None,
        "interpretation": (
            "This is a paired ablation on identical held-out valid nontrivial rows. "
            "The classical baseline uses the same sampling, nearest-parent, limits, "
            "goal region, attempts, and collision checks, but only steers directly "
            "toward each RRT sample and never calls the GAN or discriminator."
        ),
    }
    write_csv(data_dir / "testing_classical_basic_rrt_cases.csv", rows)
    save_json(data_dir / "testing_classical_basic_rrt_summary.json", summary)
    save_json(data_dir / "testing_gan_vs_classical_paired_comparison.json", comparison)
    return summary, rows, comparison


def write_metrics_and_charts(out, metrics, args):
    out = Path(out)
    master = out / "05_MODEL_AND_DATASET_FILES" / "metrics_master"
    per = master / "per_dataset_accuracy_loss"
    charts = master / "FINAL_POLISHED_CHARTS"
    mkdir(per)
    mkdir(charts)
    rows = []
    for split in ["training", "validation", "testing"]:
        row = {"dataset": split, **metrics[split]}
        rows.append(row)
        text = f"""DATASET: {split.upper()}\n\nGENERATOR:\n- Best-of full-state accuracy: {metrics[split]['generator_bestof_full_state_accuracy_percent']:.6f}%\n- Deterministic full-state accuracy: {metrics[split]['generator_deterministic_full_state_accuracy_percent']:.6f}%\n- Best-of full-state MSE loss: {metrics[split]['generator_bestof_full_state_mse_loss']:.12f}\n- Deterministic full-state MSE loss: {metrics[split]['generator_deterministic_full_state_mse_loss']:.12f}\n- Best-of position accuracy: {metrics[split]['generator_bestof_position_accuracy_percent']:.6f}%\n- Deterministic position accuracy: {metrics[split]['generator_deterministic_position_accuracy_percent']:.6f}%\n- Best-of velocity accuracy: {metrics[split]['generator_bestof_velocity_accuracy_percent']:.6f}%\n- Deterministic velocity accuracy: {metrics[split]['generator_deterministic_velocity_accuracy_percent']:.6f}%\n- Deterministic position accuracy @ 0.005: {metrics[split]['generator_deterministic_position_accuracy_at_0_005_percent']:.6f}%\n- Deterministic position accuracy @ 0.010: {metrics[split]['generator_deterministic_position_accuracy_at_0_01_percent']:.6f}%\n- Deterministic velocity accuracy @ 0.025: {metrics[split]['generator_deterministic_velocity_accuracy_at_0_025_percent']:.6f}%\n- Deterministic velocity accuracy @ 0.050: {metrics[split]['generator_deterministic_velocity_accuracy_at_0_05_percent']:.6f}%\n- Best-of mean full-state error: {metrics[split]['generator_bestof_mean_full_state_error']:.12f}\n- Deterministic mean full-state error: {metrics[split]['generator_deterministic_mean_full_state_error']:.12f}\n- Best-of p95 full-state error: {metrics[split]['generator_bestof_p95_full_state_error']:.12f}\n- Deterministic p95 full-state error: {metrics[split]['generator_deterministic_p95_full_state_error']:.12f}\n- Mean random-candidate deviation from deterministic state: {metrics[split]['mean_random_candidate_deviation_from_deterministic_state']:.12f}\n\nIMPORTANT METRIC INTERPRETATION:\nBest-of is an oracle diagnostic: it uses the true label to identify the closest candidate after generation. Deterministic accuracy is the honest single-output deployment metric. The candidate-deviation value reveals whether noise actually creates distinct choices or whether the generator has collapsed to one output. Exact normalized and physical-unit threshold definitions are in METRIC_DEFINITIONS.json.\n\nDISCRIMINATOR:\n- Final discriminator accuracy: {metrics[split]['final_discriminator_accuracy_percent']:.6f}%\n- Final discriminator BCE loss: {metrics[split]['final_discriminator_bce_loss']:.12f}\nA discriminator near 50% can be consistent with a strong generator; higher discriminator accuracy is not automatically better GAN performance.\n\nV7 CONDITION AND OUTPUT:\nCondition: {CONDITION_SCHEMA}\nGenerator output: {GENERATOR_OUTPUT_SCHEMA}\nFull state: [parent_x+delta_x,parent_y+delta_y,next_vx,next_vy].\nThe network predicts the two-component one-step velocity increment. Position and next velocity are then generated together by the exact constant-acceleration dynamics, so displacement error is never amplified by 2/dt to create velocity.\n"""
        (per / f"{split}_accuracy_and_loss.txt").write_text(text, encoding="utf-8")
    write_csv(per / "00_per_dataset_accuracy_and_loss_summary.csv", rows)
    labels = [r["dataset"] for r in rows]

    def vals(key):
        return [float(r[key]) for r in rows]

    x = np.arange(len(labels), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    series = [
        ("generator_deterministic_full_state_accuracy_percent", "Deterministic full-state", "#2E86AB"),
        ("generator_deterministic_position_accuracy_percent", "Deterministic position", "#2CA58D"),
        ("generator_deterministic_velocity_accuracy_percent", "Deterministic velocity", "#F18F01"),
    ]
    for series_index, (key, label, color) in enumerate(series):
        values = vals(key)
        bars = ax.bar(x + (series_index - 1) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("V7 deployable deterministic generator accuracy by split")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, [label.title() for label in labels])
    ax.set_ylim(0, 105)
    ax.axhline(
        args.target_accuracy, color="black", linestyle="--", linewidth=1.2,
        label=f"Requested {args.target_accuracy:g}% threshold",
    )
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(charts / "01_accuracy_grouped_bar_chart.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3))
    generator_loss = vals("generator_deterministic_full_state_mse_loss")
    discriminator_loss = vals("final_discriminator_bce_loss")
    axes[0].bar(x, generator_loss, color="#2E86AB")
    axes[0].set_title("Deterministic generator full-state MSE")
    axes[0].set_ylabel("Mean squared error")
    axes[0].set_xticks(x, [label.title() for label in labels])
    axes[0].grid(axis="y", alpha=0.30)
    axes[1].bar(x, discriminator_loss, color="#2CA58D")
    axes[1].set_title("Final discriminator BCE")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].set_xticks(x, [label.title() for label in labels])
    axes[1].grid(axis="y", alpha=0.30)
    fig.suptitle("V7 RRT-Sample GAN losses by split (separate scales)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(charts / "02_loss_separate_scales.png", dpi=180)
    plt.close(fig)

    report = ["V7 STRICT RRT-SAMPLE GAN METRICS", ""]
    for row in rows:
        report.append(row["dataset"].upper())
        report.append(f"- Generator best-of full-state accuracy: {row['generator_bestof_full_state_accuracy_percent']:.6f}%")
        report.append(f"- Generator deterministic full-state accuracy: {row['generator_deterministic_full_state_accuracy_percent']:.6f}%")
        report.append(f"- Generator deterministic position accuracy: {row['generator_deterministic_position_accuracy_percent']:.6f}%")
        report.append(f"- Generator deterministic velocity accuracy: {row['generator_deterministic_velocity_accuracy_percent']:.6f}%")
        report.append(f"- Generator best-of full-state MSE loss: {row['generator_bestof_full_state_mse_loss']:.12f}")
        report.append(f"- Generator deterministic full-state MSE loss: {row['generator_deterministic_full_state_mse_loss']:.12f}")
        report.append(f"- Generator best-of position accuracy: {row['generator_bestof_position_accuracy_percent']:.6f}%")
        report.append(f"- Generator best-of velocity accuracy: {row['generator_bestof_velocity_accuracy_percent']:.6f}%")
        report.append(f"- Final discriminator accuracy: {row['final_discriminator_accuracy_percent']:.6f}%")
        report.append(f"- Final discriminator BCE loss: {row['final_discriminator_bce_loss']:.12f}")
        report.append("")
    (out / "FINAL_METRICS_TRIPLE_CHECK.txt").write_text("\n".join(report), encoding="utf-8")


def write_training_progression_charts(out, history, args):
    charts = Path(out) / "05_MODEL_AND_DATASET_FILES" / "metrics_master" / "TRAINING_PROGRESSION_CHARTS"
    mkdir(charts)
    if not history:
        (charts / "NO_HISTORY.txt").write_text("No evaluation history was available.\n", encoding="utf-8")
        return
    epochs = np.asarray([row["epoch"] for row in history], dtype=float)

    def gap_aware(key):
        """Insert NaNs so skipped early-stop epochs are never drawn as training."""
        xs = []
        ys = []
        max_contiguous_gap = max(1.5, 1.5 * float(args.eval_every))
        for i, row in enumerate(history):
            epoch = float(row["epoch"])
            if i and epoch - float(history[i - 1]["epoch"]) > max_contiguous_gap:
                xs.append(float("nan"))
                ys.append(float("nan"))
            xs.append(epoch)
            ys.append(float(row[key]))
        return np.asarray(xs), np.asarray(ys)

    selected_row = max(history, key=lambda row: float(row["selection_score"]))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    plot_x, plot_y = gap_aware("generator_deterministic_full_state_accuracy_percent")
    ax.plot(plot_x, plot_y, linewidth=2.4, marker="o", markersize=3, label="Deterministic full-state")
    best_x, best_y = gap_aware("generator_bestof_full_state_accuracy_percent")
    ax.plot(best_x, best_y, linewidth=2.2, marker="o", markersize=3, label="Oracle best-of full-state")
    ax.scatter(
        [selected_row["epoch"]],
        [selected_row["generator_deterministic_full_state_accuracy_percent"]],
        marker="*", s=150, color="crimson", zorder=6,
        label=f"Selected checkpoint (epoch {selected_row['epoch']})",
    )
    ax.axhline(args.target_accuracy, color="black", linestyle="--", linewidth=1.2, label=f"Requested {args.target_accuracy:g}%")
    ax.axvline(args.supervised_epochs, color="gray", linestyle=":", linewidth=1.4, label="Adversarial phase begins")
    ax.set_title("Validation full-state accuracy during training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(charts / "01_validation_accuracy_progression.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    pos_x, pos_y = gap_aware("generator_deterministic_position_accuracy_percent")
    ax.plot(
        pos_x, pos_y,
        linewidth=2.3, marker="o", markersize=3, label="Deterministic position",
    )
    vel_x, vel_y = gap_aware("generator_deterministic_velocity_accuracy_percent")
    ax.plot(
        vel_x, vel_y,
        linewidth=2.3, marker="o", markersize=3, label="Deterministic velocity",
    )
    ax.axhline(
        args.target_accuracy, color="black", linestyle="--", linewidth=1.2,
        label=f"Requested {args.target_accuracy:g}%",
    )
    ax.set_title("Validation position and velocity accuracy during training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(charts / "02_validation_position_velocity_accuracy.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    mean_x, mean_y = gap_aware("generator_deterministic_mean_full_state_error")
    p95_x, p95_y = gap_aware("generator_deterministic_p95_full_state_error")
    axes[0].plot(mean_x, mean_y, linewidth=2.3, label="Mean error")
    axes[0].plot(p95_x, p95_y, linewidth=2.3, label="95th-percentile error")
    axes[0].axhline(args.state_accuracy_threshold, color="black", linestyle="--", linewidth=1.1, label="Accuracy threshold")
    axes[0].set_title("Deterministic full-state error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Euclidean state error")
    axes[0].grid(True, alpha=0.30)
    axes[0].legend(loc="best")
    loss_x, loss_y = gap_aware("train_g_loss")
    positive_g = np.where(np.isnan(loss_y), np.nan, np.maximum(loss_y, 1e-12))
    axes[1].semilogy(loss_x, positive_g, linewidth=2.3, color="#F18F01")
    axes[1].set_title("Training generator objective")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Composite generator loss (log scale)")
    axes[1].grid(True, alpha=0.30)
    fig.suptitle("Generator training progression")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(charts / "03_generator_error_and_loss_progression.png", dpi=190)
    plt.close(fig)

    adversarial_rows = [row for row in history if row["phase"] == "adversarial"]
    if adversarial_rows:
        adv_epochs = [row["epoch"] for row in adversarial_rows]
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        axes[0].plot(adv_epochs, [row["adversarial_discriminator_accuracy_percent"] for row in adversarial_rows], marker="o", linewidth=2.2)
        axes[0].axhline(50.0, color="gray", linestyle="--", linewidth=1.1, label="Chance level")
        axes[0].set_title("Adversarial discriminator accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, alpha=0.30)
        axes[0].legend(loc="best")
        axes[1].plot(adv_epochs, [row["adversarial_discriminator_bce_loss"] for row in adversarial_rows], marker="o", linewidth=2.2, color="#2CA58D")
        axes[1].set_title("Adversarial discriminator BCE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Binary cross-entropy")
        axes[1].grid(True, alpha=0.30)
        fig.suptitle("Adversarial fine-tuning progression")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(charts / "04_adversarial_progression.png", dpi=190)
        plt.close(fig)

    (charts / "README.txt").write_text(
        "Each point is a seeded validation evaluation recorded at the configured eval interval. "
        "Best-of is an oracle diagnostic; deterministic accuracy is the single-output metric. "
        "NaN breaks prevent skipped early-stopping epochs from being drawn as continuous training. "
        f"The validation-selected checkpoint is epoch {selected_row['epoch']}.\n",
        encoding="utf-8",
    )


def write_final_discriminator_progression_charts(out, rows, final_report):
    charts = Path(out) / "05_MODEL_AND_DATASET_FILES" / "metrics_master" / "FINAL_DISCRIMINATOR_PROGRESSION_CHARTS"
    mkdir(charts)
    if not rows:
        (charts / "NO_HISTORY.txt").write_text("No final-discriminator history was available.\n", encoding="utf-8")
        return
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3))
    axes[0].plot(steps, [row["training_accuracy"] for row in rows], marker="o", linewidth=2.1, label="Training")
    axes[0].plot(steps, [row["validation_accuracy"] for row in rows], marker="o", linewidth=2.1, label="Validation")
    axes[0].set_title("Classifier accuracy")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.30)
    axes[0].legend(loc="best")
    axes[1].plot(steps, [row["training_loss"] for row in rows], marker="o", linewidth=2.1, label="Training")
    axes[1].plot(steps, [row["validation_loss"] for row in rows], marker="o", linewidth=2.1, label="Validation")
    axes[1].set_title("Classifier binary cross-entropy")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("BCE")
    axes[1].grid(True, alpha=0.30)
    axes[1].legend(loc="best")
    fig.suptitle("Frozen-generator final discriminator progression")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(charts / "01_final_discriminator_progression.png", dpi=190)
    plt.close(fig)
    (charts / "README.txt").write_text(
        "Checkpoint selection used training and validation only. Testing was evaluated after checkpoint selection. "
        f"Held-out testing accuracy: {final_report.get('testing_accuracy')}%.\n"
        "In GAN interpretation, higher discriminator accuracy is not automatically better generator quality.\n",
        encoding="utf-8",
    )


def write_rollout_benchmark(
    out, rows_by_split, summaries, classical_baseline_summary=None,
):
    overall = Path(out) / "04_OVERALL_BENCHMARK"
    charts = overall / "01_OVERALL_BENCHMARK_CHARTS"
    data_dir = overall / "02_OVERALL_BENCHMARK_DATA"
    mkdir(charts)
    mkdir(data_dir)
    buckets = ["easy", "medium", "hard"]
    splits = ["training", "validation", "testing"]
    colors = ["#2E86AB", "#F18F01", "#2CA58D"]

    flat = []
    for split in splits:
        for bucket in buckets:
            flat.append({
                "split": split,
                "difficulty": bucket,
                **summaries[split]["by_difficulty"][bucket],
            })
    write_csv(data_dir / "rollout_summary_by_split_and_difficulty.csv", flat)
    write_csv(data_dir / "all_rollout_cases.csv", [r for s in splits for r in rows_by_split[s]])

    reason_rows = []
    for split in splits:
        for bucket in buckets:
            group = [r for r in rows_by_split[split] if r["difficulty_bucket"] == bucket]
            for reason in sorted(set(r["reason"] for r in group)):
                count = sum(1 for r in group if r["reason"] == reason)
                reason_rows.append({
                    "split": split,
                    "difficulty": bucket,
                    "reason": reason,
                    "count": count,
                    "percent": float(100.0 * count / max(1, len(group))),
                })
    write_csv(data_dir / "termination_reasons.csv", reason_rows)

    def grouped_chart(
        metric, ylabel, title, filename, ylim=None, annotate=True,
        confidence_interval_keys=None, annotation_format=".2f",
    ):
        x = np.arange(len(buckets), dtype=float)
        width = 0.24
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        for split_index, split in enumerate(splits):
            raw_values = [summaries[split]["by_difficulty"][b].get(metric) for b in buckets]
            values = [np.nan if value is None else float(value) for value in raw_values]
            positions = x + (split_index - 1) * width
            bars = ax.bar(positions, values, width, label=split.title(), color=colors[split_index])
            if confidence_interval_keys is not None:
                low_key, high_key = confidence_interval_keys
                lows = [
                    summaries[split]["by_difficulty"][b].get(low_key)
                    for b in buckets
                ]
                highs = [
                    summaries[split]["by_difficulty"][b].get(high_key)
                    for b in buckets
                ]
                lower_error = [
                    0.0 if low is None or not np.isfinite(value)
                    else max(0.0, value - float(low))
                    for value, low in zip(values, lows)
                ]
                upper_error = [
                    0.0 if high is None or not np.isfinite(value)
                    else max(0.0, float(high) - value)
                    for value, high in zip(values, highs)
                ]
                ax.errorbar(
                    positions, values, yerr=[lower_error, upper_error],
                    fmt="none", ecolor="black", elinewidth=1.0,
                    capsize=3, capthick=1.0,
                )
            if annotate:
                for bar, value in zip(bars, values):
                    if np.isfinite(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                            format(value, annotation_format),
                            ha="center", va="bottom", fontsize=7.5, rotation=45,
                        )
        ax.set_xticks(x, [b.title() for b in buckets])
        ax.set_xlabel("Difficulty")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.30)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=190)
        plt.close(fig)

    grouped_chart(
        "solve_rate_percent", "Valid nontrivial solve rate (%)",
        "V7 solve rate (environment-cluster bootstrap 95% intervals)",
        "01_rollout_solve_rate.png", ylim=(0, 105),
        annotation_format=".1f",
        confidence_interval_keys=(
            "solve_rate_environment_cluster_bootstrap_95_low_percent",
            "solve_rate_environment_cluster_bootstrap_95_high_percent",
        ),
    )
    grouped_chart(
        "mean_solved_path_cost", "Mean Euclidean path length (solved cases only)",
        "Solved-path length by split and difficulty", "02_rollout_solved_path_length.png",
        annotation_format=".4f",
    )
    grouped_chart(
        "mean_runtime_seconds", "Mean runtime (seconds)",
        "Planner runtime by split and difficulty", "03_rollout_runtime.png",
        annotation_format=".4f",
    )
    grouped_chart(
        "mean_nodes", "Mean total tree nodes (including root)",
        "Tree size by split and difficulty", "04_rollout_tree_nodes.png",
        annotation_format=".1f",
    )
    grouped_chart(
        "mean_best_distance_to_goal", "Mean final distance to goal",
        "Final goal distance by split and difficulty", "05_rollout_goal_distance.png",
        annotation_format=".4f",
    )
    grouped_chart(
        "gan_share_of_accepted_nodes_percent", "GAN share of accepted nodes (%)",
        "Audited learned-steering contribution by split and difficulty",
        "06_gan_share_of_accepted_nodes.png", ylim=(0, 105),
        annotation_format=".1f",
    )
    if classical_baseline_summary is not None:
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        x = np.arange(len(buckets), dtype=float)
        width = 0.34
        learned_values = [
            summaries["testing"]["by_difficulty"][bucket]["solve_rate_percent"]
            for bucket in buckets
        ]
        classical_values = [
            classical_baseline_summary["by_difficulty"][bucket]["solve_rate_percent"]
            for bucket in buckets
        ]
        learned_bars = ax.bar(
            x - width / 2, learned_values, width,
            label="GAN steering + conditional safety fallback", color="#2E86AB",
        )
        classical_bars = ax.bar(
            x + width / 2, classical_values, width,
            label="Classical direct steering", color="#8C8C8C",
        )
        for positions, source_summary, values in (
            (x - width / 2, summaries["testing"], learned_values),
            (x + width / 2, classical_baseline_summary, classical_values),
        ):
            lows = [
                source_summary["by_difficulty"][bucket][
                    "solve_rate_environment_cluster_bootstrap_95_low_percent"
                ] for bucket in buckets
            ]
            highs = [
                source_summary["by_difficulty"][bucket][
                    "solve_rate_environment_cluster_bootstrap_95_high_percent"
                ] for bucket in buckets
            ]
            ax.errorbar(
                positions, values,
                yerr=[
                    [
                        0.0 if low is None else max(0.0, value - low)
                        for value, low in zip(values, lows)
                    ],
                    [
                        0.0 if high is None else max(0.0, high - value)
                        for value, high in zip(values, highs)
                    ],
                ],
                fmt="none", ecolor="black", elinewidth=1.0,
                capsize=3, capthick=1.0,
            )
        for bars, values in (
            (learned_bars, learned_values),
            (classical_bars, classical_values),
        ):
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=8,
                )
        ax.set_title(
            "Held-out paired solve rate (environment-cluster bootstrap intervals)"
        )
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Valid nontrivial solve rate (%)")
        ax.set_xticks(x, [bucket.title() for bucket in buckets])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.30)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(charts / "07_testing_gan_vs_classical_solve_rate.png", dpi=190)
        plt.close(fig)

    explanation = [
        "V7 STRICT RRT-SAMPLE GAN BENCHMARK EXPLANATION",
        "",
        "Solve rate is the percentage of valid, nontrivial cases that reached the goal region without crossing an obstacle.",
        "Invalid starts, invalid goals, and cases already within the goal radius are excluded before difficulty bucketing and are reported in benchmark_eligibility_diagnostics.json.",
        "Each solve-rate value includes an environment-cluster bootstrap 95% interval as the primary uncertainty estimate. A row-level Wilson interval is retained only as a transparent unclustered diagnostic.",
        "Solved-path length is accumulated Euclidean distance, not the original LQR objective. It is computed only for solved cases; mixing failed partial paths into this average would be misleading.",
        "Runtime is planner computation time across all attempts for a case; image rendering time is excluded.",
        "Tree/path counters describe the selected attempt; separate total_* columns record effort accumulated across every executed attempt.",
        "Tree nodes include the root plus accepted branches, while rejected collision/boundary proposals are tracked separately in the case CSV.",
        "Runtime contracts verify one free-space sample, one Euclidean-nearest parent, the same GAN target, at most one accepted child, and zero accepted obstacle or configuration-space-boundary violations per iteration.",
        "GAN and fallback acceptances are counted separately. Fallback steering is eligible only when no GAN proposal survives all steering, safety, progress, and duplicate checks.",
        "Final goal distance is especially useful for failed cases because it shows whether failure occurred near the goal or in a true dead end.",
        "Easy, medium, and hard are relative thirds of each split according to start-goal distance, direct-path clearance, and obstacle count.",
        "The ranking score is distance + 4*max(0, 0.25 - direct_path_clearance) + 0.06*obstacle_count; it ranks cases but is not a physical cost.",
        "The held-out classical comparison uses identical testing rows and seeds, but replaces GAN/discriminator steering with one direct bounded extension toward each sample.",
        "",
        "Exact counts and values are in 02_OVERALL_BENCHMARK_DATA; the PNG files are visual summaries of those same values.",
    ]
    (overall / "00_README_OVERALL_BENCHMARK.txt").write_text(
        "\n".join(explanation) + "\n", encoding="utf-8"
    )


def copy_metrics_to_section(out, split, section):
    out = Path(out)
    section = Path(section)
    src = out / "05_MODEL_AND_DATASET_FILES" / "metrics_master"
    dst = section / "01_METRICS"
    mkdir(dst)
    shutil.copy2(src / "per_dataset_accuracy_loss" / "00_per_dataset_accuracy_and_loss_summary.csv", dst / "00_all_dataset_accuracy_and_loss_summary.csv")
    shutil.copy2(src / "per_dataset_accuracy_loss" / f"{split}_accuracy_and_loss.txt", dst / f"{split}_accuracy_and_loss.txt")
    (dst / "CHART_LOCATION.txt").write_text(
        "Authoritative all-split charts are stored once at "
        "05_MODEL_AND_DATASET_FILES/metrics_master/FINAL_POLISHED_CHARTS. "
        "They are not duplicated here.\n",
        encoding="utf-8",
    )


def write_explanation_files(
    out, data, diagnostics, ceiling, ambiguity, target_diagnostics, split_overlap,
    benchmark_eligibility, condition_reconstruction, args,
):
    notes = Path(out) / "06_NOTES_AND_EXPLANATIONS"
    mkdir(notes)
    (notes / "METHOD_EXPLANATION.txt").write_text(
        f"""V7.7 INVARIANT CONSTRAINT-ATTENTION RRT-SAMPLE GAN METHOD EXPLANATION

This is a basic RRT whose local steering function is learned from expert state transitions.
Every iteration uses this exact order:

1. Sample a random free-space target (or the final goal with controlled goal bias).
2. Select the existing tree node nearest to that sampled target.
3. Build the GAN condition as {CONDITION_SCHEMA}.
4. Generate local motion candidates with output {GENERATOR_OUTPUT_SCHEMA}.
5. Convert each proposal to [parent_x+delta_x,parent_y+delta_y,next_vx,next_vy].
6. Enforce bounded step, speed, turn, workspace, duplicate-state, and full-edge collision checks.
7. Score valid candidates by progress toward the sampled target, clearance, smoothness, and expert-likeness.
8. Add at most one selected node with the nearest node as its parent.
9. Declare success only when that single node enters the goal region; never append a multi-node connector chain.

The sampled target used for nearest-parent selection is exactly the sampled target given to the GAN. The GAN never continues using the final goal when RRT selected a different target.

The network's primary learned action is the two-component one-step velocity increment. The full output is reconstructed as delta_position=(parent_velocity+0.5*delta_velocity)*dt and next_velocity=clip(parent_velocity+delta_velocity). This enforces the verified expert dynamics without amplifying a learned displacement error by 2/dt.

V7.7 starts from scratch because the previous absolute-coordinate network fit training environments almost perfectly but did not generalize to held-out environments. Learned layers never receive raw absolute parent/sample coordinates or a flattened ordered obstacle map. Motion is expressed in the sample-aligned frame. Every obstacle, workspace boundary, and acceleration limit is encoded as a shared CBF/QP inequality token containing its local normal, bound, nominal-control slack, clearance, and type. Attention, mean pooling, and max pooling aggregate this token set without depending on obstacle order.

The learned output is a correction to the documented label-free nominal LQR control. A paired parity evaluation enforces reflection equivariance for arbitrary weights, while the sample frame enforces rotation equivariance. Random exact D4 square rotations/reflections are also applied to training rows. Validation and testing are never augmented. The generator is smaller and uses AdamW regularization. Aggressive hard-example oversampling is disabled by default because it previously improved training error while validation remained flat.

Validation alone selects checkpoints; testing remains held out. The 98% target is a stopping/certification criterion, not a promised or fabricated result. Final discriminator training, rollout benchmarking, pictures, and packaging remain blocked unless the selected deterministic validation metrics reach the target or the user explicitly overrides that guard.

This is basic RRT with learned steering, not RRT*. It does not claim RRT* rewiring or asymptotic optimality. Euclidean path length is reported separately from the original paper's LQR state-control objective.

Benchmark solve rates include only valid, nontrivial scenarios. Invalid starts, invalid goals, and rows already inside the goal radius are counted in a separate eligibility report and cannot lower or inflate planner success.

For the testing split, a paired classical basic-RRT ablation uses the same rows and seeds with direct sample steering and no neural inference. This measures whether learned steering improves solve rate, path length, or runtime.
""",
        encoding="utf-8",
    )
    (notes / "WHAT_CHANGED_FROM_V7_6.txt").write_text(
        """WHAT CHANGED FROM V7.6 TO V7.7

The V7.6 diagnostic isolated environment overfitting. During 300 additional
epochs, training strict-velocity accuracy stayed near 99.93% and training mean
velocity error fell from 0.00286 to 0.00050. Held-out validation strict-velocity
accuracy stayed near 77%, and the validation-selected result occurred at epoch
1. More hard mining therefore made memorization stronger without improving
generalization.

Source review found the direct shortcut: the old trunk still consumed the six
raw normalized parent/velocity/sample fields, including absolute parent and
sample coordinates, despite a comment describing the input as relative-only.
Its large ordered-obstacle MLP could also memorize obstacle arrangements.

V7.7 removes that architecture instead of fine-tuning it. It starts from random
weights, never imports V7.5/V7.6 generator weights, and exposes only relative
sample-frame motion plus physically meaningful LQR/CBF constraint tokens to
learned layers. Shared token encoding and set pooling make obstacle ordering
irrelevant. Reflection parity averaging and the sample frame enforce square
symmetry, while exact D4 training augmentation balances all orientations.

The model is substantially smaller, adds AdamW weight decay, and disables hard-
example oversampling by default. The deterministic deployable output still
controls validation selection. Testing remains untouched until finalization.
Reaching 98% remains an experimental target and is never manufactured.
""",
        encoding="utf-8",
    )
    (notes / "WHAT_CHANGED_FROM_V6.txt").write_text(
        """WHAT CHANGED IN V7

V6 trained mostly on final-goal-conditioned path transitions because the old
dataset did not retain the exact RRT random sample that caused each expansion.
V7 regenerates every supervised row at the tree insertion boundary: exact
sample, exact Euclidean-nearest parent, and exact accepted LQR-CBF child.

The old source encoded obstacle centers relative to the current state, but its
augmentation and rollout code treated them as absolute coordinates. Four of
five augmented copies were therefore geometrically wrong. V7 uses one explicit
schema—absolute normalized center_x, center_y, radius—and validates/reorders it
nearest-first at every rollout node.

V7 splits whole environments before training, refuses legacy datasets, keeps
testing out of checkpoint selection, marks best-of as oracle-only, and lets only
deterministic full-state, position, and velocity metrics certify the requested
accuracy target. The discriminator is diagnostic and cannot certify generator
quality.

The final held-out benchmark runs a paired classical basic-RRT baseline with
the same rows, samples, seeds, limits, attempts, goal region, and collision
checks. Learned-steering benefit is measured rather than assumed.
""",
        encoding="utf-8",
    )
    (notes / "REAL_WORLD_IMPACT.txt").write_text(
        """REAL WORLD IMPACT\n\nRobots need to plan safe motion through obstacles while respecting physical motion limits. Pure random search can waste samples, and pure neural prediction can be unsafe.\n\nThis project combines both approaches. The GAN proposes expert-like motions learned from strict sample-conditioned LQR-CBF basic-RRT expansions. The RRT planner checks every proposal with full-edge collision and kinodynamic constraints.\n\nPotential applications include mobile robots, robot arms, warehouse robots, drones, search-and-rescue robots, and space rovers.\n""",
        encoding="utf-8",
    )
    (notes / "LEARNING_SUMMARY.txt").write_text(
        """WHAT THIS PROJECT TAUGHT\n\nImportant concepts covered by this project:\n- motion planning\n- geometric vs kinodynamic planning\n- state representation [x,y,vx,vy]\n- RRT tree expansion\n- parent-pointer path reconstruction\n- collision checking along an edge\n- sample-conditioned LQR-CBF basic RRT as an expert teacher\n- conditional GANs\n- generator and discriminator roles\n- generator noise and candidate diversity\n- supervised pretraining vs adversarial fine tuning\n- final discriminator classifier training\n- PyTorch tensors, modules, losses, optimizers, training loops, and checkpoints\n- diagnostics for finding bottlenecks\n- why position accuracy and full-state accuracy differ\n- why honest train/validation/test separation matters\n""",
        encoding="utf-8",
    )
    save_json(notes / "dataset_diagnostics.json", diagnostics)
    save_json(notes / "representation_ceiling.json", ceiling)
    save_json(
        notes / "condition_only_expert_reconstruction_diagnostics.json",
        condition_reconstruction,
    )
    save_json(notes / "condition_ambiguity_diagnostics.json", ambiguity)
    save_json(notes / "target_condition_diagnostics.json", target_diagnostics)
    save_json(notes / "split_scenario_overlap_diagnostics.json", split_overlap)
    save_json(notes / "benchmark_eligibility_diagnostics.json", benchmark_eligibility)


def create_package(
    data, G, D, C, norm, device, args, history, final_d_report,
    final_d_log, diagnostics, ceiling, ambiguity, target_diagnostics, split_overlap,
    benchmark_eligibility, condition_reconstruction,
):
    package_started = time.perf_counter()

    def stage(message):
        print(
            f"FINALIZATION STAGE: {message} | total elapsed="
            f"{format_duration(time.perf_counter() - package_started)}",
            flush=True,
        )

    out = Path(args.output_dir)
    stage("preparing verified output directory")
    d4_report = d4_augmentation_diagnostics(data)
    if out.exists():
        if any(out.iterdir()) and not args.overwrite_output:
            raise FileExistsError(
                f"Output directory became non-empty during training: {out}. "
                "Refusing to delete it without --overwrite_output."
            )
        shutil.rmtree(out)
    mkdir(out)
    model_dir = out / "05_MODEL_AND_DATASET_FILES" / "model"
    mkdir(model_dir)
    torch.save(
        {
            "generator_state_dict": {k: v.detach().cpu() for k, v in G.state_dict().items()},
            "adversarial_discriminator_state_dict": {k: v.detach().cpu() for k, v in D.state_dict().items()},
            "final_discriminator_state_dict": {k: v.detach().cpu() for k, v in C.state_dict().items()},
            "normalizer": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm.items()},
            "noise_dim": args.noise_dim,
            "method": METHOD_NAME,
            "script_version": SCRIPT_VERSION,
            "condition_schema": CONDITION_SCHEMA,
            "generator_output": GENERATOR_OUTPUT_SCHEMA,
            "state_reconstruction": "[parent_x+delta_x,parent_y+delta_y,next_vx,next_vy]",
            "velocity_architecture": (
                "D4-equivariant sample-frame control-increment residual over "
                "a label-free nominal LQR prior; CBF/QP inequality tokens are "
                "encoded with shared weights, permutation-equivariant token "
                "interaction, and invariant attention/mean/max pooling; "
                "position and next velocity are "
                "jointly reconstructed through exact one-step dynamics"
            ),
            "generalization_training": {
                "starts_from_scratch": True,
                "legacy_generator_weights_imported": False,
                "raw_absolute_coordinates_reach_learned_layers": False,
                "obstacle_order_affects_generator": False,
                "d4_augmentation_probability": float(
                    args.d4_augmentation_probability
                ),
                "validation_or_testing_augmented": False,
                "generator_weight_decay": float(args.generator_weight_decay),
                "hard_example_fraction": float(args.hard_example_fraction),
                "hard_examples_use_training_split_only": True,
                "nominal_lqr_feature_is_label_free": True,
                "nominal_lqr_position_increment_gain": float(
                    NOMINAL_LQR_POSITION_INCREMENT_GAIN
                ),
                "nominal_lqr_velocity_increment_gain": float(
                    NOMINAL_LQR_VELOCITY_INCREMENT_GAIN
                ),
                "nominal_lqr_step_distance_normalized": float(
                    NOMINAL_LQR_STEP_DISTANCE_NORMALIZED
                ),
                "d4_numerical_audit": d4_report,
            },
            "velocity_forced_to_equal_delta_over_dt": False,
            "engineered_geometry_features": (
                "sample-frame distance and parent velocity; nominal LQR "
                "control; per-constraint local normal, normalized inequality "
                "bound, unclipped-objective slack/violation, clipped-baseline "
                "feasibility, label-free single-halfspace projection "
                "correction, clearance, radius, distance, and "
                "obstacle/boundary/acceleration type"
            ),
            "rrt_operation_order": "free-space sample -> Euclidean-nearest parent -> GAN steer -> safety check -> add at most one node",
            "goal_connection_policy": "terminate when the single accepted node enters the goal region; no multi-node connector",
            "goal_radius_normalized": float(data["goal_radius"]),
            "collision_margin_normalized": float(data["collision_margin"]),
            "supervision_lookahead_steps": int(
                data["supervision_lookahead_steps"]
            ),
            "runtime_effective_dt_normalized": float(
                args.runtime_effective_dt_normalized
            ),
            "continuous_edge_safety_schema": data[
                "continuous_edge_safety_schema"
            ],
            "geometric_connectivity_filter": data[
                "geometric_connectivity_filter"
            ],
            "expert_cbf_control_filter_schema": data[
                "cbf_control_filter_schema"
            ],
            "target_sources": data["target_source"].tolist(),
            "exact_recorded_rrt_sample_supervision": bool(all(
                source == "recorded_rrt_sample_xy"
                for source in data["target_source"].tolist()
            )),
            "no_mpc": True,
            "no_state_refiner": True,
            "gradient_updates_use_training_split_only": True,
            "validation_used_for_checkpoint_selection": True,
            "testing_held_out_from_checkpoint_selection": True,
        },
        model_dir / "v7_7_invariant_constraint_rrt_sample_gan_model.pt",
    )
    write_csv(model_dir / "training_history.csv", history)
    save_json(model_dir / "final_discriminator_report.json", final_d_report)
    write_csv(model_dir / "final_discriminator_training_log.csv", final_d_log)
    np.savez_compressed(out / "05_MODEL_AND_DATASET_FILES" / "rrt_sample_gan_dataset_v7.npz", **data)
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "dataset_diagnostics.json", diagnostics)
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "representation_ceiling.json", ceiling)
    save_json(
        out / "05_MODEL_AND_DATASET_FILES"
        / "condition_only_expert_reconstruction_diagnostics.json",
        condition_reconstruction,
    )
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "condition_ambiguity_diagnostics.json", ambiguity)
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "target_condition_diagnostics.json", target_diagnostics)
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "split_scenario_overlap_diagnostics.json", split_overlap)
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "benchmark_eligibility_diagnostics.json", benchmark_eligibility)
    save_json(out / "05_MODEL_AND_DATASET_FILES" / "run_config.json", vars(args))
    save_json(
        out / "05_MODEL_AND_DATASET_FILES" / "d4_augmentation_diagnostics.json",
        d4_report,
    )
    stage("selected model, dataset, diagnostics, and histories saved")

    metrics = {}
    metric_evaluation_started = time.perf_counter()
    metric_specs = [
        ("training", "train_X", "train_y"),
        ("validation", "validation_X", "validation_y"),
        ("testing", "testing_X", "testing_y"),
    ]
    for split_index, (split, xkey, ykey) in enumerate(metric_specs):
        gm = evaluate_generator(
            G, data[xkey], data[ykey], norm, device, args, args.eval_candidates,
            seed=args.seed + 1001 + split_index,
        )
        dm = evaluate_final_discriminator(
            C, G, data[xkey], data[ykey], norm, device, args,
            limit=None, seed=args.seed + 2001 + split_index,
        )
        metrics[split] = {**gm, **dm}
        print(
            measured_progress(
                "FINAL SPLIT METRICS",
                split_index + 1,
                len(metric_specs),
                metric_evaluation_started,
            )
            + f" | completed={split}",
            flush=True,
        )
    position_scale = float(data["position_scale"][0])
    velocity_scale = float(data["velocity_scale"])
    metric_definitions = {
        "state_order": "[normalized_x, normalized_y, normalized_vx, normalized_vy]",
        "full_state_accuracy": (
            "percentage of rows whose four-component Euclidean state error is "
            f"<= {args.state_accuracy_threshold}; mixed normalized units"
        ),
        "position_accuracy": (
            "percentage of rows whose 2D Euclidean position error is "
            f"<= {args.position_accuracy_threshold} normalized workspace units"
        ),
        "velocity_accuracy": (
            "percentage of rows whose 2D Euclidean velocity error is "
            f"<= {args.velocity_accuracy_threshold} normalized velocity units"
        ),
        "control_increment_accuracy_at_0_05": (
            "diagnostic percentage whose recovered two-component one-step "
            "normalized velocity-increment error is <= 0.05; the increment is "
            "recovered from delta_position and parent_velocity using the exact "
            "constant-acceleration equation"
        ),
        "configured_thresholds_normalized": {
            "full_state": float(args.state_accuracy_threshold),
            "position": float(args.position_accuracy_threshold),
            "velocity": float(args.velocity_accuracy_threshold),
        },
        "configured_thresholds_physical": {
            "full_state": "not a single physical unit; combines normalized position and velocity",
            "position_distance": float(args.position_accuracy_threshold * position_scale),
            "velocity_distance": float(args.velocity_accuracy_threshold * velocity_scale),
        },
        "planner_success_and_safety": {
            "goal_radius_normalized": float(args.goal_radius),
            "goal_radius_physical_distance": float(
                args.goal_radius * position_scale
            ),
            "combined_robot_and_safety_margin_normalized": float(
                args.collision_margin
            ),
            "combined_robot_and_safety_margin_physical_distance": float(
                args.collision_margin * position_scale
            ),
            "workspace_boundary_is_shrunk_by_collision_margin": True,
            "expert_continuous_edge_safety_schema": data[
                "continuous_edge_safety_schema"
            ],
            "expert_supervision_lookahead_steps": int(
                data["supervision_lookahead_steps"]
            ),
            "runtime_effective_dt_normalized": float(
                args.runtime_effective_dt_normalized
            ),
        },
        "position_scale_physical_units_per_normalized_unit": position_scale,
        "velocity_scale_physical_units_per_normalized_unit": velocity_scale,
        "mse_definition": "mean squared component error over all rows and all four state components",
        "deterministic_definition": "single zero-noise output used as the deployable prediction metric",
        "bestof_definition": (
            "oracle diagnostic: generate candidate_count outputs and use the true label "
            "afterward to select the closest; never a deployable selector"
        ),
        "additional_fixed_thresholds_normalized": {
            "full_state": [0.05, 0.10],
            "position": [0.005, 0.01, 0.025, 0.05],
            "velocity": [0.025, 0.05, 0.10],
        },
    }
    save_json(out / "FINAL_METRICS.json", metrics)
    save_json(out / "METRIC_DEFINITIONS.json", metric_definitions)
    save_json(
        out / "05_MODEL_AND_DATASET_FILES" / "metric_definitions.json",
        metric_definitions,
    )
    write_metrics_and_charts(out, metrics, args)
    write_training_progression_charts(out, history, args)
    write_final_discriminator_progression_charts(out, final_d_log, final_d_report)
    stage("held-out metrics, exact tables, and core charts saved")

    sections = {
        "training": out / "01_TRAINING_DATASET",
        "validation": out / "02_VALIDATION_DATASET",
        "testing": out / "03_TESTING_DATASET",
    }
    for split, section in sections.items():
        mkdir(section)
        (section / f"00_README_{split.upper()}_DATASET.txt").write_text(
            f"""CLEAR TITLE: {split.upper()} DATASET

Method: V7 Strict Basic RRT with Sample-Conditioned GAN Steering

Folders:
01_METRICS
02_V7_STRICT_RRT_SAMPLE_GAN_PICTURES
03_ROLLOUT_DATA

Condition: {CONDITION_SCHEMA}
Generator output: {GENERATOR_OUTPUT_SCHEMA}
Full state: [parent_x+delta_x,parent_y+delta_y,next_vx,next_vy].

The free-space RRT sample is selected before the Euclidean-nearest parent and is then passed to the GAN.
At most one accepted child is added per RRT iteration. Entering the goal region ends the search; no multi-node connector is used.
Invalid starts/goals and zero-iteration trivial cases are excluded from solve-rate benchmarks and reported separately.
No MPC, state refiner, RRT* rewiring, or one-step picture folder is claimed.
""",
            encoding="utf-8",
        )

    (out / "00_README_START_HERE.txt").write_text(
        f"""V7 STRICT RRT-SAMPLE GAN - START HERE

Package: {out.name}

Method: basic RRT with a sample-conditioned learned local steering function.

Required operation order:
valid random/goal-biased sample -> Euclidean-nearest tree parent -> GAN(parent,sample,obstacles) -> safety checks -> add at most one child.

Goal policy: the search succeeds when that one child enters the goal region. There is no hidden multi-node goal-connection chain.
Benchmark policy: invalid starts, invalid goals, and cases already inside the goal radius never enter solve-rate statistics.

Condition: {CONDITION_SCHEMA}
Generator output: {GENERATOR_OUTPUT_SCHEMA}
State reconstruction: [parent_x+delta_x,parent_y+delta_y,next_vx,next_vy].
Training target sources: {data['target_source'].tolist()}

All three sources must say recorded_rrt_sample_xy. V7 refuses to run otherwise.

Dataset rows:
training: {len(data['train_X'])}
validation: {len(data['validation_X'])}
testing: {len(data['testing_X'])}

Scientific boundaries:
- validation/testing never receive gradient updates
- testing is excluded from checkpoint selection
- best-of is an oracle diagnostic, not deployment selection
- no fake 98% claim
- no MPC or state-refiner claim
- this is basic RRT, not RRT*
- reported path cost is Euclidean path length, not the paper's LQR objective
- benchmark uncertainty primarily uses a 2,000-replicate environment-cluster bootstrap; the unclustered row-level Wilson interval is retained only as a diagnostic
- exactly {args.benchmark_cases_per_difficulty} valid nontrivial cases are evaluated per difficulty and split
- unless explicitly skipped, held-out testing rows are rerun with a paired classical direct-steering basic-RRT baseline
""",
        encoding="utf-8",
    )

    summaries = {}
    rollout_rows = {}
    summaries["training"], rollout_rows["training"] = generate_split_visuals(
        "training", data["train_X"], data["training_goal_state"],
        data["train_environment_id"], data["train_scenario_id"],
        G, C, norm, device, args, sections["training"], seed=args.seed + 100,
    )
    copy_metrics_to_section(out, "training", sections["training"])
    summaries["validation"], rollout_rows["validation"] = generate_split_visuals(
        "validation", data["validation_X"], data["validation_goal_state"],
        data["validation_environment_id"], data["validation_scenario_id"],
        G, C, norm, device, args, sections["validation"], seed=args.seed + 200,
    )
    copy_metrics_to_section(out, "validation", sections["validation"])
    summaries["testing"], rollout_rows["testing"] = generate_split_visuals(
        "testing", data["testing_X"], data["testing_goal_state"],
        data["testing_environment_id"], data["testing_scenario_id"],
        G, C, norm, device, args, sections["testing"], seed=args.seed + 300,
    )
    copy_metrics_to_section(out, "testing", sections["testing"])
    stage("training, validation, and testing GAN-RRT benchmarks completed")

    classical_summary = None
    classical_rows = []
    classical_comparison = None
    overall = out / "04_OVERALL_BENCHMARK"
    if not args.skip_classical_baseline:
        classical_summary, classical_rows, classical_comparison = (
            generate_classical_testing_baseline(
                data["testing_X"], data["testing_goal_state"],
                rollout_rows["testing"], args, overall, seed=args.seed + 300,
            )
        )
    stage("paired classical-RRT benchmark completed")

    mkdir(overall / "02_OVERALL_BENCHMARK_DATA")
    save_json(
        overall / "02_OVERALL_BENCHMARK_DATA" / "benchmark_eligibility_diagnostics.json",
        benchmark_eligibility,
    )
    write_csv(
        overall / "02_OVERALL_BENCHMARK_DATA" / "rollout_summary_by_split.csv",
        [{k: v for k, v in {"split": split, **summary}.items() if k != "by_difficulty"} for split, summary in summaries.items()],
    )
    write_rollout_benchmark(
        out, rollout_rows, summaries,
        classical_baseline_summary=classical_summary,
    )

    write_explanation_files(
        out, data, diagnostics, ceiling, ambiguity, target_diagnostics, split_overlap,
        benchmark_eligibility, condition_reconstruction, args,
    )
    save_json(
        out / "06_NOTES_AND_EXPLANATIONS" / "d4_augmentation_diagnostics.json",
        d4_report,
    )
    stage("benchmark charts and scientific explanations saved")

    repro = out / "07_CODE_AND_REPRODUCIBILITY"
    mkdir(repro)
    source_root = Path(__file__).resolve().parent
    source_suite = repro / "SOURCE_SUITE"
    mkdir(source_suite)
    source_names = [
        "build_v7_7_rrt_sample_gan.py",
        "generate_v7_rrt_sample_dataset.py",
        "render_v7_7_path_pictures.py",
        "build_v7_7_results_gallery.py",
        "verify_v7_7_final_package.py",
        "run_v7_7_final_pipeline.sh",
        "monitor_v7_7_training.py",
        "v7_7.sh",
        "README_V7_7.txt",
        "COPY_PASTE_COMMANDS.txt",
        "AUDIT_REPORT_V7_7.txt",
        "SOURCE_MANIFEST.sha256",
    ]
    copied_sources = []
    source_copy_errors = []
    for source_name in source_names:
        source_path = source_root / source_name
        if not source_path.is_file():
            source_copy_errors.append(f"missing source file: {source_path}")
            continue
        destination = source_suite / source_name
        try:
            shutil.copy2(source_path, destination)
            copied_sources.append({
                "file": source_name,
                "sha256": sha256_file(destination),
                "size_bytes": int(destination.stat().st_size),
            })
        except Exception as exc:
            source_copy_errors.append(f"{source_name}: {exc}")
    save_json(
        repro / "SOURCE_SUITE_INVENTORY.json",
        {
            "files": copied_sources,
            "errors": source_copy_errors,
            "all_requested_sources_copied": not source_copy_errors,
        },
    )
    trainer_copy = source_suite / "build_v7_7_rrt_sample_gan.py"
    if trainer_copy.is_file():
        shutil.copy2(trainer_copy, repro / "build_v7_7_rrt_sample_gan.py")

    selected_training_row = max(
        history, key=lambda row: float(row["selection_score"])
    )
    save_json(
        model_dir / "VALIDATION_SELECTED_CHECKPOINT_SUMMARY.json",
        {
            "selected_epoch": int(selected_training_row["epoch"]),
            "selected_phase": selected_training_row["phase"],
            "selection_score": float(selected_training_row["selection_score"]),
            "deterministic_position_accuracy_percent": float(
                selected_training_row[
                    "generator_deterministic_position_accuracy_percent"
                ]
            ),
            "deterministic_velocity_accuracy_percent": float(
                selected_training_row[
                    "generator_deterministic_velocity_accuracy_percent"
                ]
            ),
            "deterministic_full_state_accuracy_percent": float(
                selected_training_row[
                    "generator_deterministic_full_state_accuracy_percent"
                ]
            ),
            "deployable_validation_target_reached": bool(
                selected_training_row["deployable_validation_target_reached"]
            ),
            "packaged_model_file": (
                "v7_7_invariant_constraint_rrt_sample_gan_model.pt"
            ),
        },
    )
    checkpoint_sources = [
        (
            training_checkpoint_path(args),
            model_dir / "V7_7_FINAL_TRAINING_CHECKPOINT.pt",
        ),
        (
            final_discriminator_checkpoint_path(args),
            model_dir / "V7_7_FINAL_DISCRIMINATOR_CHECKPOINT.pt",
        ),
    ]
    checkpoint_inventory = []
    for checkpoint_source, checkpoint_destination in checkpoint_sources:
        if checkpoint_source.is_file():
            shutil.copy2(checkpoint_source, checkpoint_destination)
            checkpoint_inventory.append({
                "file": checkpoint_destination.name,
                "sha256": sha256_file(checkpoint_destination),
                "size_bytes": int(checkpoint_destination.stat().st_size),
            })
    save_json(model_dir / "CHECKPOINT_INVENTORY.json", checkpoint_inventory)
    (repro / "RUN_COMMAND_USED.txt").write_text(
        shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    runtime_environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "selected_device": str(device),
    }
    save_json(repro / "runtime_environment.json", runtime_environment)
    (repro / "requirements_versions_used.txt").write_text(
        f"numpy=={np.__version__}\n"
        f"matplotlib=={matplotlib.__version__}\n"
        f"torch=={torch.__version__}\n",
        encoding="utf-8",
    )
    (repro / "NO_CHEATING_RULES.txt").write_text("No validation/testing training. No fake metrics. Report target flags exactly as verification_report.json states.\n", encoding="utf-8")
    stage("complete source suite and resumable checkpoints preserved")

    bad = []
    for p in out.rglob("*"):
        s = str(p.relative_to(out)).lower()
        if "one_step" in s or "__pycache__" in s or "_temp" in s:
            bad.append(str(p.relative_to(out)))
    required_relative_paths = [
        "00_README_START_HERE.txt",
        "FINAL_METRICS.json",
        "METRIC_DEFINITIONS.json",
        "FINAL_METRICS_TRIPLE_CHECK.txt",
        "04_OVERALL_BENCHMARK/01_OVERALL_BENCHMARK_CHARTS/01_rollout_solve_rate.png",
        "04_OVERALL_BENCHMARK/01_OVERALL_BENCHMARK_CHARTS/06_gan_share_of_accepted_nodes.png",
        "04_OVERALL_BENCHMARK/02_OVERALL_BENCHMARK_DATA/all_rollout_cases.csv",
        "05_MODEL_AND_DATASET_FILES/model/v7_7_invariant_constraint_rrt_sample_gan_model.pt",
        "05_MODEL_AND_DATASET_FILES/model/training_history.csv",
        "06_NOTES_AND_EXPLANATIONS/METHOD_EXPLANATION.txt",
        "07_CODE_AND_REPRODUCIBILITY/build_v7_7_rrt_sample_gan.py",
        "07_CODE_AND_REPRODUCIBILITY/SOURCE_SUITE_INVENTORY.json",
        "07_CODE_AND_REPRODUCIBILITY/SOURCE_SUITE/v7_7.sh",
        "05_MODEL_AND_DATASET_FILES/model/VALIDATION_SELECTED_CHECKPOINT_SUMMARY.json",
        "05_MODEL_AND_DATASET_FILES/model/V7_7_FINAL_TRAINING_CHECKPOINT.pt",
        "07_CODE_AND_REPRODUCIBILITY/runtime_environment.json",
    ]
    if not args.skip_classical_baseline:
        required_relative_paths.extend([
            "04_OVERALL_BENCHMARK/01_OVERALL_BENCHMARK_CHARTS/07_testing_gan_vs_classical_solve_rate.png",
            "04_OVERALL_BENCHMARK/02_OVERALL_BENCHMARK_DATA/testing_classical_basic_rrt_cases.csv",
            "04_OVERALL_BENCHMARK/02_OVERALL_BENCHMARK_DATA/testing_gan_vs_classical_paired_comparison.json",
        ])
    missing_required_paths = [
        relative for relative in required_relative_paths
        if not (out / relative).is_file()
    ]
    min_bestof = min(metrics[s]["generator_bestof_full_state_accuracy_percent"] for s in ["training", "validation", "testing"])
    min_det = min(metrics[s]["generator_deterministic_full_state_accuracy_percent"] for s in ["training", "validation", "testing"])
    min_det_position = min(
        metrics[s]["generator_deterministic_position_accuracy_percent"]
        for s in ["training", "validation", "testing"]
    )
    min_det_velocity = min(
        metrics[s]["generator_deterministic_velocity_accuracy_percent"]
        for s in ["training", "validation", "testing"]
    )
    min_disc = min(metrics[s]["final_discriminator_accuracy_percent"] for s in ["training", "validation", "testing"])
    min_ceiling = min(ceiling[s]["ceiling_full_state_accuracy_0_10_percent"] for s in ["training", "validation", "testing"])
    all_rollout_rows = [
        row for split_rows in rollout_rows.values() for row in split_rows
    ]
    rrt_contract_passed = bool(all_rollout_rows) and all(
        bool(row["runtime_contract_passed"])
        and row["sampled_target_count"] == row["nearest_parent_selection_count"]
        and row["sampled_target_count"] == row["gan_condition_count"]
        and row["sample_condition_mismatch_count"] == 0
        and row["max_children_added_in_any_iteration"] <= 1
        and row["accepted_total"] <= row["sampled_target_count"]
        and row["accepted_total"]
        == row["gan_accepted_total"] + row["fallback_accepted_total"]
        and row["goal_connector_nodes"] == 0
        and row["accepted_edge_collision_violations"] == 0
        for row in all_rollout_rows
    )
    expected_cases_per_split = 3 * int(args.benchmark_cases_per_difficulty)
    benchmark_counts_passed = all(
        len(rollout_rows[split]) == expected_cases_per_split
        and all(
            sum(row["difficulty_bucket"] == bucket for row in rollout_rows[split])
            == args.benchmark_cases_per_difficulty
            for bucket in ("easy", "medium", "hard")
        )
        for split in ("training", "validation", "testing")
    )
    benchmark_eligibility_passed = all(
        row["scenario_eligibility"] == "valid_nontrivial"
        for row in all_rollout_rows
    )
    classical_baseline_passed = bool(args.skip_classical_baseline) or bool(
        len(classical_rows) == len(rollout_rows["testing"])
        and all(bool(row["runtime_contract_passed"]) for row in classical_rows)
        and all(
            int(row["accepted_edge_collision_violations"]) == 0
            for row in classical_rows
        )
        and {
            int(row["row_index"]) for row in classical_rows
        } == {
            int(row["row_index"]) for row in rollout_rows["testing"]
        }
    )
    package_valid = bool(
        len(bad) == 0
        and len(missing_required_paths) == 0
        and rrt_contract_passed
        and benchmark_counts_passed
        and benchmark_eligibility_passed
        and classical_baseline_passed
        and args.children_per_expand == 1
    )
    # Only deployable generator metrics determine the requested-accuracy status.
    # Oracle best-of uses the true label for selection, and discriminator
    # accuracy is not monotonic GAN quality, so neither can certify success.
    all_requested_targets = bool(
        min_det >= args.target_accuracy
        and min_det_position >= args.target_accuracy
        and min_det_velocity >= args.target_accuracy
    )
    verification = {
        "status": (
            "targets_reached" if package_valid and all_requested_targets
            else "completed_below_target" if package_valid
            else "package_invalid"
        ),
        "package_structure_valid": package_valid,
        "all_requested_targets_reached": all_requested_targets,
        "package": str(out),
        "method": METHOD_NAME,
        "script_version": SCRIPT_VERSION,
        "condition_schema": CONDITION_SCHEMA,
        "generator_output": GENERATOR_OUTPUT_SCHEMA,
        "state_reconstruction": "[parent_x+delta_x,parent_y+delta_y,next_vx,next_vy]",
        "rrt_sample_selected_before_parent": bool(rrt_contract_passed),
        "nearest_parent_selected_before_gan": bool(rrt_contract_passed),
        "same_sample_passed_to_gan": bool(rrt_contract_passed),
        "one_accepted_child_per_iteration": bool(rrt_contract_passed),
        "rollout_sample_parent_count_contract_passed": bool(rrt_contract_passed),
        "runtime_rrt_contract_passed": bool(rrt_contract_passed),
        "multi_node_goal_connector_enabled": False,
        "goal_connector_nodes_across_benchmark": int(
            sum(row["goal_connector_nodes"] for row in all_rollout_rows)
        ),
        "accepted_edge_collision_violations_across_benchmark": int(
            sum(row["accepted_edge_collision_violations"] for row in all_rollout_rows)
        ),
        "gan_accepted_nodes_across_benchmark": int(
            sum(row["gan_accepted_total"] for row in all_rollout_rows)
        ),
        "fallback_accepted_nodes_across_benchmark": int(
            sum(row["fallback_accepted_total"] for row in all_rollout_rows)
        ),
        "nonfinite_candidates_safely_rejected_across_benchmark": int(
            sum(row["nonfinite_candidate_rejections"] for row in all_rollout_rows)
        ),
        "gan_share_of_accepted_nodes_across_benchmark_percent": float(
            100.0
            * sum(row["gan_accepted_total"] for row in all_rollout_rows)
            / max(1, sum(row["accepted_total"] for row in all_rollout_rows))
        ),
        "benchmark_counts_passed": bool(benchmark_counts_passed),
        "benchmark_eligibility_passed": bool(benchmark_eligibility_passed),
        "classical_testing_baseline_skipped": bool(args.skip_classical_baseline),
        "classical_testing_baseline_contract_passed": bool(classical_baseline_passed),
        "testing_gan_vs_classical_paired_comparison": classical_comparison,
        "benchmark_cases_per_difficulty_per_split": int(args.benchmark_cases_per_difficulty),
        "benchmark_excludes_invalid_and_trivial_scenarios": True,
        "benchmark_unique_physical_scenario_signatures": {
            split: int(summaries[split]["unique_physical_scenario_signatures"])
            for split in ("training", "validation", "testing")
        },
        "wilson_interval_unit": "benchmark row",
        "wilson_interval_independence_warning": (
            "Rows from the same physical goal/obstacle scenario can be correlated. "
            "The row-level Wilson intervals are diagnostic only and are not cluster-adjusted."
        ),
        "primary_solve_rate_interval": (
            "2,000-replicate percentile bootstrap resampling whole environment_id clusters"
        ),
        "benchmark_eligibility_diagnostics": benchmark_eligibility,
        "rrt_star_rewiring_claimed": False,
        "velocity_output_contains_next_vx_next_vy": True,
        "velocity_uses_exact_double_integrator_initial_prior": False,
        "velocity_is_generated_from_primary_control_increment": True,
        "control_increment_equation": (
            "delta_position=(parent_velocity+0.5*delta_velocity)*dt; "
            "next_velocity=clip(parent_velocity+delta_velocity)"
        ),
        "control_increment_has_two_learned_components": True,
        "position_and_velocity_share_one_dynamics_consistent_action": True,
        "velocity_forced_to_equal_delta_over_dt": False,
        "target_sources": data["target_source"].tolist(),
        "exact_recorded_rrt_sample_supervision": bool(all(
            source == "recorded_rrt_sample_xy"
            for source in data["target_source"].tolist()
        )),
        "target_condition_diagnostics": target_diagnostics,
        "split_scenario_overlap_diagnostics": split_overlap,
        "representation_ceiling_min_full_state_accuracy_percent": float(min_ceiling),
        "no_mpc": True,
        "no_state_refiner": True,
        "gradient_updates_use_training_split_only": True,
        "validation_used_for_checkpoint_selection": True,
        "testing_held_out_from_checkpoint_selection": True,
        "no_one_step_pictures": len(bad) == 0,
        "bad_paths": bad,
        "missing_required_paths": missing_required_paths,
        "training_rows": int(len(data["train_X"])),
        "validation_rows": int(len(data["validation_X"])),
        "testing_rows": int(len(data["testing_X"])),
        "minimum_generator_bestof_full_state_accuracy_percent": float(min_bestof),
        "minimum_generator_deterministic_full_state_accuracy_percent": float(min_det),
        "minimum_generator_deterministic_position_accuracy_percent": float(min_det_position),
        "minimum_generator_deterministic_velocity_accuracy_percent": float(min_det_velocity),
        "minimum_final_discriminator_accuracy_percent": float(min_disc),
        "testing_generator_deterministic_full_state_accuracy_percent": float(
            metrics["testing"]["generator_deterministic_full_state_accuracy_percent"]
        ),
        "testing_generator_deterministic_position_accuracy_percent": float(
            metrics["testing"]["generator_deterministic_position_accuracy_percent"]
        ),
        "testing_generator_deterministic_velocity_accuracy_percent": float(
            metrics["testing"]["generator_deterministic_velocity_accuracy_percent"]
        ),
        "testing_rollout_solve_rate_percent": float(summaries["testing"]["solve_rate_percent"]),
        "testing_rollout_solve_rate_wilson_95_low_percent": summaries["testing"]["solve_rate_wilson_95_low_percent"],
        "testing_rollout_solve_rate_wilson_95_high_percent": summaries["testing"]["solve_rate_wilson_95_high_percent"],
        "testing_rollout_solve_rate_environment_cluster_bootstrap_95_low_percent": summaries["testing"]["solve_rate_environment_cluster_bootstrap_95_low_percent"],
        "testing_rollout_solve_rate_environment_cluster_bootstrap_95_high_percent": summaries["testing"]["solve_rate_environment_cluster_bootstrap_95_high_percent"],
        "testing_valid_nontrivial_rollout_cases": int(summaries["testing"]["cases"]),
        "target_accuracy_percent": float(args.target_accuracy),
        "generator_bestof_target_reached_oracle_diagnostic_only": bool(
            min_bestof >= args.target_accuracy
        ),
        "generator_deterministic_target_reached": bool(min_det >= args.target_accuracy),
        "generator_deterministic_position_target_reached": bool(
            min_det_position >= args.target_accuracy
        ),
        "generator_deterministic_velocity_target_reached": bool(
            min_det_velocity >= args.target_accuracy
        ),
        "discriminator_reached_same_numeric_threshold_diagnostic_only": bool(
            min_disc >= args.target_accuracy
        ),
        "discriminator_interpretation_warning": (
            "For a GAN, discriminator accuracy near 50% can indicate that generated and real samples are hard to distinguish. "
            "Discriminator accuracy is reported but is excluded from the generator target gate."
        ),
        "training_visual_summary": summaries["training"],
        "validation_visual_summary": summaries["validation"],
        "testing_visual_summary": summaries["testing"],
        "created_unix": int(time.time()),
    }
    save_json(out / "verification_report.json", verification)
    if not verification["package_structure_valid"]:
        print(json.dumps(verification, indent=2), flush=True)
        raise RuntimeError("Package verification failed.")
    (out / "BUILD_COMPLETED_SUCCESSFULLY.txt").write_text(
        "V7 BUILD COMPLETED SUCCESSFULLY\n\n"
        f"Status: {verification['status']}\n"
        f"Script version: {SCRIPT_VERSION}\n"
        f"Package: {out}\n"
        f"ZIP: {Path(str(out) + '.zip')}\n"
        f"Exact recorded RRT-sample supervision: {verification['exact_recorded_rrt_sample_supervision']}\n"
        "Authoritative statistics: verification_report.json and FINAL_METRICS.json\n"
        "Target status uses deterministic full-state, position, and velocity accuracy across all three splits. "
        "Oracle best-of and discriminator accuracy cannot certify target attainment.\n"
        "A completed_below_target status means execution succeeded but one or more deployable accuracy targets were not honestly reached.\n",
        encoding="utf-8",
    )
    manifest = {}
    for p in out.rglob("*"):
        if p.is_file():
            manifest[str(p.relative_to(out))] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    save_json(out / "FILE_MANIFEST_SHA256.json", manifest)
    stage("package structure, planner contract, and file inventory verified")
    zip_path = Path(str(out) + ".zip")
    if zip_path.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"ZIP appeared during training: {zip_path}. "
                "Refusing to replace it without --overwrite_output."
            )
        zip_path.unlink()
    shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    stage("final ZIP archive created")
    print(json.dumps(verification, indent=2), flush=True)
    print("BUILD FINISHED SUCCESSFULLY", flush=True)
    print("FINAL STATUS:", verification["status"], flush=True)
    print(
        "TESTING DETERMINISTIC ACCURACY: "
        f"state={verification['testing_generator_deterministic_full_state_accuracy_percent']:.6f}% "
        f"position={verification['testing_generator_deterministic_position_accuracy_percent']:.6f}% "
        f"velocity={verification['testing_generator_deterministic_velocity_accuracy_percent']:.6f}%",
        flush=True,
    )
    cluster_low = verification[
        "testing_rollout_solve_rate_environment_cluster_bootstrap_95_low_percent"
    ]
    cluster_high = verification[
        "testing_rollout_solve_rate_environment_cluster_bootstrap_95_high_percent"
    ]
    cluster_text = (
        f"environment-cluster bootstrap 95% interval {cluster_low:.6f}% to "
        f"{cluster_high:.6f}%"
        if cluster_low is not None and cluster_high is not None
        else "environment-cluster interval unavailable (fewer than two clusters)"
    )
    print(
        "TESTING VALID-NONTRIVIAL SOLVE RATE: "
        f"{verification['testing_rollout_solve_rate_percent']:.6f}% "
        f"({cluster_text})",
        flush=True,
    )
    print(
        "GAN SHARE OF ACCEPTED BENCHMARK NODES: "
        f"{verification['gan_share_of_accepted_nodes_across_benchmark_percent']:.6f}%",
        flush=True,
    )
    if classical_comparison is not None:
        print(
            "PAIRED TESTING SOLVE RATE (GAN / CLASSICAL): "
            f"{classical_comparison['gan_solve_rate_percent']:.6f}% / "
            f"{classical_comparison['classical_solve_rate_percent']:.6f}%",
            flush=True,
        )
    print("ZIP:", zip_path, flush=True)


def validate_args(args):
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name} must be finite")
    positive_ints = [
        "batch_size", "noise_dim", "eval_every", "eval_candidates", "final_d_steps",
        "eval_candidate_chunk", "d_steps",
        "final_d_eval_every", "max_iterations", "max_nodes", "candidates_per_expand",
        "children_per_expand", "attempts_per_case",
        "min_attempts_before_accept",
        "max_rejected_edges_stored",
        "benchmark_cases_per_difficulty",
        "hard_mining_refresh_epochs",
        "generalization_probe_rows",
    ]
    for name in positive_ints:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name} must be at least 1")
    if args.max_nodes < 2:
        raise ValueError("--max_nodes must be at least 2 (root plus one possible child)")
    if args.supervised_epochs < 0 or args.adversarial_epochs < 0:
        raise ValueError("Epoch counts cannot be negative")
    if args.supervised_epochs + args.adversarial_epochs < 1:
        raise ValueError("At least one training epoch is required")
    for name in ["lr_g", "lr_d", "final_d_lr", "nominal_dt"]:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not 0.0 <= args.lr_decay_start_fraction < 1.0:
        raise ValueError("--lr_decay_start_fraction must be in [0, 1)")
    if not 0.0 < args.lr_final_factor <= 1.0:
        raise ValueError("--lr_final_factor must be in (0, 1]")
    if not 0.0 < args.adversarial_lr_factor <= 1.0:
        raise ValueError("--adversarial_lr_factor must be in (0, 1]")
    if args.generator_weight_decay < 0.0:
        raise ValueError("--generator_weight_decay cannot be negative")
    if not 0.0 <= args.d4_augmentation_probability <= 1.0:
        raise ValueError(
            "--d4_augmentation_probability must be between 0 and 1"
        )
    for name in ["train_noise_scale", "eval_noise_scale", "planner_noise_scale", "final_d_fake_noise_scale"]:
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name} cannot be negative")
    for name in [
        "goal_sample_rate", "final_d_zero_noise_fraction",
        "state_threshold_margin_fraction",
        "velocity_threshold_margin_fraction", "position_threshold_margin_fraction",
    ]:
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be between 0 and 1")
    for name in ["tail_fraction", "velocity_tail_fraction"]:
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name} must be in (0, 1]")
    if args.min_step <= 0 or args.max_step < args.min_step:
        raise ValueError("Require 0 < --min_step <= --max_step")
    if args.min_speed < 0 or args.max_speed < args.min_speed:
        raise ValueError("Require 0 <= --min_speed <= --max_speed")
    if not 0.0 < args.max_turn_deg <= 180.0:
        raise ValueError("--max_turn_deg must be in (0, 180]")
    if args.goal_radius <= 0 or args.collision_margin < 0:
        raise ValueError("Goal radius must be positive and collision margin cannot be negative")
    if args.goal_connect_distance < 0:
        raise ValueError("--goal_connect_distance cannot be negative")
    for name in ["state_accuracy_threshold", "position_accuracy_threshold", "velocity_accuracy_threshold"]:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not 0.0 <= args.target_accuracy <= 100.0:
        raise ValueError("--target_accuracy must be between 0 and 100")
    if not 0.0 <= args.hard_example_fraction <= 4.0:
        raise ValueError("--hard_example_fraction must be between 0 and 4")
    if abs(float(args.hard_example_fraction)) > 1e-12:
        raise ValueError(
            "V7.7 requires --hard_example_fraction 0. The V7.6 experiment "
            "proved that extra hard-example passes drove training error toward "
            "zero without improving held-out environment accuracy."
        )
    if args.hard_priority_start_ratio < 0.0:
        raise ValueError("--hard_priority_start_ratio cannot be negative")
    if args.hard_priority_strength < 0.0:
        raise ValueError("--hard_priority_strength cannot be negative")
    if args.hard_priority_power <= 0.0:
        raise ValueError("--hard_priority_power must be positive")
    if args.hard_priority_cap <= 0.0:
        raise ValueError("--hard_priority_cap must be positive")
    if args.children_per_expand != 1:
        raise ValueError("V7 basic RRT requires --children_per_expand 1 (at most one accepted node per random sample)")
    if args.random_fallback_candidates < 0:
        raise ValueError("--random_fallback_candidates cannot be negative")
    if args.sample_progress_weight <= 0 or args.progress_weight < 0:
        raise ValueError("Sample-progress weights must be valid")
    if args.detour_clearance_gain < 0:
        raise ValueError("--detour_clearance_gain cannot be negative")
    if args.sample_rejection_limit < 1:
        raise ValueError("--sample_rejection_limit must be at least 1")
    if args.workspace_max <= args.workspace_min:
        raise ValueError("--workspace_max must be greater than --workspace_min")
    if 2.0 * args.collision_margin >= args.workspace_max - args.workspace_min:
        raise ValueError("Collision margin leaves no free interior workspace")
    if abs(args.workspace_min) > 1e-12 or abs(args.workspace_max - 1.0) > 1e-12:
        raise ValueError("This dataset format uses normalized workspace bounds [0,1]")
    if abs(args.nearest_velocity_weight) > 1e-12:
        raise ValueError("Strict basic RRT requires --nearest_velocity_weight 0 for Euclidean-nearest parent selection")
    if args.duplicate_position_resolution <= 0 or args.duplicate_velocity_resolution <= 0:
        raise ValueError("Duplicate-state resolutions must be positive")
    if args.min_attempts_before_accept > args.attempts_per_case:
        raise ValueError("--min_attempts_before_accept cannot exceed --attempts_per_case")
    if args.pictures_per_split < 0:
        raise ValueError("--pictures_per_split cannot be negative")
    if args.minimum_supervised_epochs < 0:
        raise ValueError("--minimum_supervised_epochs cannot be negative")
    if args.minimum_supervised_epochs > args.supervised_epochs:
        raise ValueError("--minimum_supervised_epochs cannot exceed --supervised_epochs")
    if args.residual_only_epochs != 0:
        raise ValueError(
            "V7.7 requires --residual_only_epochs 0 because its invariant "
            "architecture is trained completely from scratch"
        )
    if args.resume and args.restart_training:
        raise ValueError("--resume and --restart_training are mutually exclusive")
    if args.warm_start_checkpoint:
        raise ValueError(
            "V7.7 intentionally rejects warm starts. The V7.5/V7.6 weights "
            "encode the absolute-coordinate shortcut being removed. Start a "
            "new V7.7 output, or use --resume only for an existing V7.7 run."
        )
    if args.early_stopping_patience_evaluations < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and minimum delta cannot be negative")
    nonnegative_weights = [
        "param_mse_weight", "state_mse_weight", "state_huber_weight",
        "direction_weight", "step_weight", "speed_weight", "progress_weight",
        "state_threshold_weight", "tail_state_weight", "adv_weight",
        "velocity_huber_weight", "velocity_direction_weight",
        "velocity_magnitude_weight", "velocity_threshold_weight",
        "velocity_tail_weight", "position_threshold_weight",
        "kinematic_consistency_weight", "stochastic_supervision_weight",
        "control_increment_mse_weight", "control_increment_huber_weight",
        "control_increment_threshold_weight", "control_increment_tail_weight",
    ]
    for name in nonnegative_weights:
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name} cannot be negative")
    supervised_weight_names = [
        name for name in nonnegative_weights
        if name not in {"adv_weight", "stochastic_supervision_weight"}
    ]
    if sum(float(getattr(args, name)) for name in supervised_weight_names) <= 0:
        raise ValueError("At least one supervised generator-loss weight must be positive")
    requested_output = Path(args.output_dir).expanduser()
    if requested_output.is_symlink():
        raise ValueError(
            f"--output_dir must not be a symbolic link: {requested_output}"
        )
    output = requested_output.resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    forbidden = {Path("/").resolve(), HOME.resolve()}
    if output in forbidden:
        raise ValueError(f"Unsafe --output_dir: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"--output_dir exists but is not a directory: {output}")
    if (
        not args.preflight_only
        and output.exists()
        and any(output.iterdir())
        and not args.overwrite_output
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new --output_dir, "
            "or explicitly add --overwrite_output."
        )
    if output == dataset or output in dataset.parents:
        raise ValueError(f"Unsafe --output_dir would contain/delete the dataset: {output}")
    derived_files = {
        Path(str(output) + ".zip"),
        Path(str(output) + "_TRAINING_CHECKPOINT.pt"),
        Path(str(output) + "_FINAL_DISCRIMINATOR_CHECKPOINT.pt"),
    }
    if dataset in derived_files:
        raise ValueError(
            "Unsafe output naming would overwrite the input dataset with a ZIP or checkpoint"
        )
    for derived in derived_files:
        if derived.exists() and not derived.is_file():
            raise ValueError(
                f"Expected output ZIP/checkpoint path is not a regular file: {derived}"
            )
    output_zip = Path(str(output) + ".zip")
    if (
        not args.preflight_only
        and output_zip.exists()
        and not args.overwrite_output
    ):
        raise FileExistsError(
            f"Output ZIP already exists: {output_zip}. Choose a new --output_dir, "
            "or explicitly add --overwrite_output."
        )
    checkpoint_files = {
        Path(str(output) + "_TRAINING_CHECKPOINT.pt"),
        Path(str(output) + "_FINAL_DISCRIMINATOR_CHECKPOINT.pt"),
    }
    if args.warm_start_checkpoint:
        warm_start_resolved = Path(args.warm_start_checkpoint).expanduser().resolve()
        if warm_start_resolved in checkpoint_files:
            raise ValueError(
                "Warm-start source must not be the new experiment's output checkpoint"
            )
    existing_checkpoints = sorted(
        str(path) for path in checkpoint_files if path.exists()
    )
    if (
        not args.preflight_only
        and existing_checkpoints
        and not args.resume
        and not args.restart_training
    ):
        raise FileExistsError(
            "Checkpoint files already exist. Use --resume with the exact original "
            "arguments, choose a new --output_dir, or explicitly add "
            f"--restart_training. Existing files: {existing_checkpoints}"
        )
    source_file = Path(__file__).resolve()
    if output == source_file or output in source_file.parents:
        raise ValueError(
            f"Unsafe --output_dir would contain/delete the running script: {output}"
        )
    existing_parent = output.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir() or not os.access(
        existing_parent, os.W_OK | os.X_OK
    ):
        raise ValueError(
            f"Output parent is not writable/searchable: {existing_parent}"
        )
    free_bytes = shutil.disk_usage(existing_parent).free
    if free_bytes < 250 * 1024 * 1024:
        raise ValueError(
            f"Less than 250 MiB is free at the output location: {existing_parent}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(HOME / "V7_RRT_SAMPLE_DATASET" / "rrt_sample_gan_dataset_v7.npz"),
    )
    parser.add_argument(
        "--output_dir", default=str(HOME / "V7_7_GENERALIZATION_GAN")
    )
    parser.add_argument("--supervised_epochs", type=int, default=400)
    parser.add_argument("--adversarial_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--noise_dim", type=int, default=16)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=4e-5)
    parser.add_argument("--lr_decay_start_fraction", type=float, default=0.50)
    parser.add_argument("--lr_final_factor", type=float, default=0.05)
    parser.add_argument(
        "--generator_weight_decay", type=float, default=1e-4,
        help="AdamW regularization for the invariant generator.",
    )
    parser.add_argument(
        "--d4_augmentation_probability", type=float, default=1.0,
        help=(
            "Probability of applying a random exact square rotation/reflection "
            "to each training row; validation/testing are never augmented."
        ),
    )
    parser.add_argument("--adversarial_lr_factor", type=float, default=0.25)
    parser.add_argument("--param_mse_weight", type=float, default=1.5)
    parser.add_argument("--state_mse_weight", type=float, default=7.0)
    parser.add_argument("--state_huber_weight", type=float, default=2.0)
    parser.add_argument("--direction_weight", type=float, default=0.70)
    parser.add_argument("--step_weight", type=float, default=20.0)
    parser.add_argument("--speed_weight", type=float, default=24.0)
    parser.add_argument("--velocity_huber_weight", type=float, default=8.0)
    parser.add_argument("--velocity_direction_weight", type=float, default=1.5)
    parser.add_argument("--velocity_magnitude_weight", type=float, default=12.0)
    parser.add_argument("--velocity_threshold_weight", type=float, default=180.0)
    parser.add_argument("--velocity_tail_weight", type=float, default=50.0)
    parser.add_argument("--velocity_tail_fraction", type=float, default=0.35)
    parser.add_argument("--control_increment_mse_weight", type=float, default=6.0)
    parser.add_argument("--control_increment_huber_weight", type=float, default=2.0)
    parser.add_argument("--control_increment_threshold_weight", type=float, default=180.0)
    parser.add_argument("--control_increment_tail_weight", type=float, default=50.0)
    parser.add_argument("--velocity_threshold_margin_fraction", type=float, default=0.50)
    parser.add_argument("--position_threshold_weight", type=float, default=10.0)
    parser.add_argument("--position_threshold_margin_fraction", type=float, default=0.70)
    parser.add_argument("--kinematic_consistency_weight", type=float, default=2.0)
    parser.add_argument("--stochastic_supervision_weight", type=float, default=0.25)
    parser.add_argument("--progress_weight", type=float, default=4.0)
    parser.add_argument("--state_threshold_weight", type=float, default=100.0)
    parser.add_argument("--state_threshold_margin_fraction", type=float, default=0.50)
    parser.add_argument("--tail_state_weight", type=float, default=25.0)
    parser.add_argument("--tail_fraction", type=float, default=0.35)
    parser.add_argument("--adv_weight", type=float, default=0.0004)
    parser.add_argument("--d_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_candidates", type=int, default=16)
    parser.add_argument("--eval_candidate_chunk", type=int, default=16)
    parser.add_argument(
        "--generalization_probe_rows", type=int, default=6000,
        help=(
            "Fixed evenly spaced training rows evaluated deterministically at "
            "each validation checkpoint to expose train/validation overfit."
        ),
    )
    parser.add_argument("--minimum_supervised_epochs", type=int, default=80)
    parser.add_argument("--early_stopping_patience_evaluations", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.02)
    parser.add_argument("--state_accuracy_threshold", type=float, default=0.10)
    parser.add_argument("--position_accuracy_threshold", type=float, default=0.05)
    parser.add_argument("--velocity_accuracy_threshold", type=float, default=0.05)
    parser.add_argument("--target_accuracy", type=float, default=98.0)
    parser.add_argument(
        "--warm_start_checkpoint",
        default="",
        help=(
            "Reserved for same-architecture experiments. V7.7 rejects legacy "
            "V7.5/V7.6 warm starts and should normally start from scratch."
        ),
    )
    parser.add_argument(
        "--hard_example_fraction", type=float, default=0.0,
        help=(
            "Optional additional training-only hard examples. Keep at zero "
            "for the generalization experiment; V7.6 already proved that "
            "aggressive hard mining worsens environment overfit."
        ),
    )
    parser.add_argument("--hard_mining_refresh_epochs", type=int, default=5)
    parser.add_argument(
        "--residual_only_epochs", type=int, default=0,
        help=(
            "Legacy compatibility option; V7.7 requires zero and trains its "
            "entire invariant architecture from scratch."
        ),
    )
    parser.add_argument("--hard_priority_start_ratio", type=float, default=0.45)
    parser.add_argument("--hard_priority_strength", type=float, default=8.0)
    parser.add_argument("--hard_priority_power", type=float, default=2.0)
    parser.add_argument("--hard_priority_cap", type=float, default=6.0)
    parser.add_argument("--train_noise_scale", type=float, default=0.025)
    parser.add_argument("--eval_noise_scale", type=float, default=0.08)
    parser.add_argument("--planner_noise_scale", type=float, default=0.12)
    parser.add_argument("--final_d_steps", type=int, default=3000)
    parser.add_argument("--final_d_lr", type=float, default=7e-4)
    parser.add_argument("--final_d_eval_every", type=int, default=150)
    parser.add_argument("--final_d_fake_noise_scale", type=float, default=0.12)
    parser.add_argument("--final_d_zero_noise_fraction", type=float, default=0.25)
    parser.add_argument(
        "--pictures_per_split", type=int, default=0,
        help="Number of rollout path pictures per split (zero keeps training/finalization image-free).",
    )
    parser.add_argument("--benchmark_cases_per_difficulty", type=int, default=100)
    parser.add_argument("--max_iterations", type=int, default=4500)
    parser.add_argument("--max_nodes", type=int, default=900)
    parser.add_argument("--candidates_per_expand", type=int, default=64)
    parser.add_argument("--children_per_expand", type=int, default=1)
    parser.add_argument("--random_fallback_candidates", type=int, default=6)
    parser.add_argument("--goal_sample_rate", type=float, default=0.18)
    parser.add_argument("--sample_rejection_limit", type=int, default=64)
    parser.add_argument("--nearest_velocity_weight", type=float, default=0.0)
    parser.add_argument("--sample_progress_weight", type=float, default=7.0)
    parser.add_argument("--workspace_min", type=float, default=0.0)
    parser.add_argument("--workspace_max", type=float, default=1.0)
    parser.add_argument("--max_rejected_edges_stored", type=int, default=2000)
    parser.add_argument("--duplicate_position_resolution", type=float, default=0.0005)
    parser.add_argument("--duplicate_velocity_resolution", type=float, default=0.01)
    parser.add_argument("--attempts_per_case", type=int, default=3)
    parser.add_argument("--min_attempts_before_accept", type=int, default=1)
    parser.add_argument("--goal_radius", type=float, default=0.0405)
    parser.add_argument(
        "--goal_connect_distance", type=float, default=0.0,
        help="Deprecated compatibility option; V7 never adds a multi-node goal connector.",
    )
    parser.add_argument("--min_progress", type=float, default=-0.08)
    parser.add_argument("--detour_clearance_gain", type=float, default=0.002)
    parser.add_argument("--min_step", type=float, default=0.0001)
    parser.add_argument("--max_step", type=float, default=0.016)
    parser.add_argument("--nominal_dt", type=float, default=0.12)
    parser.add_argument("--min_speed", type=float, default=0.0)
    parser.add_argument("--max_speed", type=float, default=1.0)
    parser.add_argument("--max_turn_deg", type=float, default=50.0)
    parser.add_argument("--collision_margin", type=float, default=0.016)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument(
        "--stop_after_training",
        action="store_true",
        help=(
            "Stop after the validation-selected GAN checkpoint is trained; "
            "resume later without this flag to run final classification, "
            "rollout benchmarks, charts, and packaging."
        ),
    )
    parser.add_argument(
        "--allow_below_target_finalization",
        action="store_true",
        help=(
            "Explicitly allow discriminator evaluation, rollout benchmarks, "
            "and packaging when the validation-selected deployable generator "
            "is still below --target_accuracy. By default V7.7 blocks this "
            "expensive work."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--restart_training", action="store_true",
        help="Explicitly discard same-output V7 training checkpoints and start from epoch 1.",
    )
    parser.add_argument(
        "--overwrite_output", action="store_true",
        help="Explicitly replace an existing non-empty output directory and ZIP after training.",
    )
    parser.add_argument(
        "--skip_classical_baseline", action="store_true",
        help="Skip the paired held-out classical basic-RRT ablation (not recommended for final reporting).",
    )
    args = parser.parse_args()
    args.script_version = SCRIPT_VERSION
    print("SCRIPT VERSION:", SCRIPT_VERSION, flush=True)
    validate_args(args)
    args.dataset = str(Path(args.dataset).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.warm_start_checkpoint:
        args.warm_start_checkpoint = str(
            Path(args.warm_start_checkpoint).expanduser().resolve()
        )
    if args.goal_connect_distance > 0:
        print(
            "WARNING: --goal_connect_distance is deprecated and ignored. "
            "V7 enforces at most one accepted node per iteration and terminates in the goal region.",
            flush=True,
        )
    args.max_turn_rad = math.radians(args.max_turn_deg)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data = load_dataset(args.dataset)
    validate_nominal_lqr_feature_contract(data)
    d4_report = d4_augmentation_diagnostics(data)
    print(
        "D4 AUGMENTATION DIAGNOSTICS:",
        json.dumps(d4_report, indent=2),
        flush=True,
    )
    if data["condition_schema"] != CONDITION_SCHEMA:
        raise RuntimeError(
            f"Dataset condition schema {data['condition_schema']!r} does not match "
            f"trainer schema {CONDITION_SCHEMA!r}"
        )
    if not math.isclose(args.workspace_min, data["workspace_min"], abs_tol=1e-8):
        raise RuntimeError("--workspace_min disagrees with dataset metadata")
    if not math.isclose(args.workspace_max, data["workspace_max"], abs_tol=1e-8):
        raise RuntimeError("--workspace_max disagrees with dataset metadata")
    if args.collision_margin + 1e-9 < data["collision_margin"]:
        raise RuntimeError(
            f"--collision_margin={args.collision_margin} is smaller than the expert "
            f"configuration-space margin {data['collision_margin']}"
        )
    if not math.isclose(
        args.nominal_dt, data["nominal_dt"], rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError(
            f"--nominal_dt={args.nominal_dt} disagrees with dataset dt={data['nominal_dt']}"
        )
    if not math.isclose(
        args.goal_radius, data["goal_radius"], rel_tol=0.0, abs_tol=1e-7
    ):
        raise RuntimeError(
            f"--goal_radius={args.goal_radius} disagrees with the expert dataset "
            f"goal radius={data['goal_radius']}. Evaluation may not loosen the "
            "expert success criterion."
        )
    # Convert the physical duration represented by a fixed-horizon child into
    # normalized-position / normalized-velocity units. Runtime fallback and the
    # classical baseline must use this value, not the one-step physical dt.
    args.runtime_effective_dt_normalized = float(
        data["supervision_lookahead_steps"]
        * data["nominal_dt"]
        * data["velocity_scale"]
        / float(data["position_scale"][0])
    )
    if args.runtime_effective_dt_normalized <= 0.0:
        raise RuntimeError("Derived runtime effective interval must be positive")
    print(
        "RUNTIME EFFECTIVE LOCAL-EDGE INTERVAL (NORMALIZED):",
        f"{args.runtime_effective_dt_normalized:.9f}",
        flush=True,
    )
    args.dataset_sha256 = sha256_file(Path(args.dataset).expanduser())
    print("DATASET SHA256:", args.dataset_sha256, flush=True)
    print("DATASET:", flush=True)
    print("training", data["train_X"].shape, data["train_y"].shape, flush=True)
    print("validation", data["validation_X"].shape, data["validation_y"].shape, flush=True)
    print("testing", data["testing_X"].shape, data["testing_y"].shape, flush=True)
    print("CONDITION SCHEMA:", CONDITION_SCHEMA, flush=True)
    print("GENERATOR OUTPUT SCHEMA:", GENERATOR_OUTPUT_SCHEMA, flush=True)
    print("TARGET SOURCES:", data["target_source"].tolist(), flush=True)
    exact_recorded_rrt_supervision = all(
        source == "recorded_rrt_sample_xy"
        for source in data["target_source"].tolist()
    )
    if not exact_recorded_rrt_supervision:
        raise RuntimeError("Strict V7 sample supervision unexpectedly failed after loading")
    target_diagnostics = target_condition_diagnostics(
        data, include_testing=False
    )
    print("TARGET-CONDITION DIAGNOSTICS:", json.dumps(target_diagnostics, indent=2), flush=True)
    if target_diagnostics["training"]["positive_progress_percent"] < 50.0:
        print(
            "WARNING: fewer than half of expert transitions move closer to their conditioned target. "
            "This dataset may contain many intentional detours or may not be suitable for sample-conditioned steering.",
            flush=True,
        )
    split_overlap = split_scenario_overlap_diagnostics(data)
    print("SPLIT SCENARIO OVERLAP DIAGNOSTICS:", json.dumps(split_overlap, indent=2), flush=True)
    if split_overlap["any_overlap"]:
        print(
            "WARNING: at least one final-goal/obstacle scenario signature occurs in more than one split. "
            "For publication-quality evaluation, regenerate the split by whole scenario.",
            flush=True,
        )
    diagnostics = dataset_diagnostics(data, include_testing=False)
    print("DATASET DIAGNOSTICS:", json.dumps(diagnostics, indent=2), flush=True)
    worst_dynamics_reconstruction_error = max(
        row["max_exact_dynamics_velocity_reconstruction_error"]
        for row in diagnostics
    )
    if worst_dynamics_reconstruction_error > 5e-5:
        raise RuntimeError(
            "Dataset labels do not obey the required one-step double-integrator "
            "equation closely enough for the control-consistent representation: "
            f"maximum normalized velocity reconstruction error="
            f"{worst_dynamics_reconstruction_error:.9g}"
        )
    train_params = target_direction_step_speed(data["train_X"], data["train_y"])
    pre_norm = fit_normalizer(
        data["train_X"],
        train_params,
        motion_dt=args.runtime_effective_dt_normalized,
    )
    ceiling = representation_ceiling(
        data, pre_norm,
        state_threshold=args.state_accuracy_threshold,
        position_threshold=args.position_accuracy_threshold,
        velocity_threshold=args.velocity_accuracy_threshold,
        include_testing=False,
    )
    print("REPRESENTATION CEILING:", json.dumps(ceiling, indent=2), flush=True)
    minimum_ceiling = min(
        ceiling[split]["ceiling_full_state_accuracy_0_10_percent"]
        for split in ("training", "validation")
    )
    if minimum_ceiling < args.target_accuracy:
        print(
            f"WARNING: the lossless local-motion representation ceiling is {minimum_ceiling:.6f}%, "
            f"below the requested {args.target_accuracy:.6f}%. Training cannot honestly exceed its representation ceiling.",
            flush=True,
        )
    condition_reconstruction = condition_only_expert_reconstruction_diagnostics(
        data, args
    )
    print(
        "CONDITION-ONLY EXPERT RECONSTRUCTION DIAGNOSTICS:",
        json.dumps(condition_reconstruction, indent=2),
        flush=True,
    )
    condition_ceiling = min(
        condition_reconstruction[split][metric]
        for split in ("training", "validation")
        for metric in (
            "condition_only_position_accuracy_percent",
            "condition_only_velocity_accuracy_percent",
            "condition_only_full_state_accuracy_percent",
        )
    )
    failed_condition_reconstructions = sum(
        condition_reconstruction[split]["failed_qp_reconstructions"]
        for split in ("training", "validation")
    )
    if (
        condition_ceiling < args.target_accuracy
        or failed_condition_reconstructions > 0
    ):
        raise RuntimeError(
            "Exact teacher reconstruction from the encoded training/validation "
            f"conditions reached only {condition_ceiling:.6f}% with "
            f"{failed_condition_reconstructions} failed QP reconstructions. "
            "V7.7 refuses to spend time on a 98% neural target until the "
            "condition-sufficiency problem is fixed."
        )
    ambiguity = condition_ambiguity_diagnostics(
        data,
        pre_norm,
        conflict_threshold=min(
            args.state_accuracy_threshold,
            args.position_accuracy_threshold,
            args.velocity_accuracy_threshold,
        ),
        include_testing=False,
    )
    print("CONDITION AMBIGUITY DIAGNOSTICS:", json.dumps(ambiguity, indent=2), flush=True)
    training_conflict_percent = ambiguity["training"]["conflicting_row_percent"]
    if training_conflict_percent > 0.0:
        print(
            f"WARNING: {training_conflict_percent:.6f}% of training rows fall in rounded-condition groups "
            "with conflicting targets. This can limit deterministic exact-state accuracy.",
            flush=True,
        )
    benchmark_eligibility = benchmark_eligibility_diagnostics(data, args)
    print(
        "BENCHMARK ELIGIBILITY DIAGNOSTICS:",
        json.dumps(benchmark_eligibility, indent=2),
        flush=True,
    )
    required_valid = 3 * args.benchmark_cases_per_difficulty
    for split in ("training", "validation", "testing"):
        available = benchmark_eligibility[split]["valid_nontrivial"]
        if available < required_valid:
            raise RuntimeError(
                f"{split}: {available} valid nontrivial scenarios are available, but "
                f"{required_valid} are required for {args.benchmark_cases_per_difficulty} "
                "cases in each of easy, medium, and hard."
            )
    selected_device = "cpu" if args.cpu or not torch.cuda.is_available() else torch.cuda.get_device_name(0)
    print("SELECTED DEVICE:", selected_device, flush=True)
    smoke_device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    run_model_smoke_test(data, pre_norm, args, smoke_device)
    run_warm_start_smoke_test(data, pre_norm, args, smoke_device)
    if args.preflight_only:
        print("PREFLIGHT PASSED: dataset, arguments, imports, and diagnostics are valid.", flush=True)
        return
    if args.restart_training:
        for checkpoint_path in (
            training_checkpoint_path(args),
            final_discriminator_checkpoint_path(args),
        ):
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"REMOVED OLD CHECKPOINT FOR EXPLICIT RESTART: {checkpoint_path}", flush=True)
    G, D, norm, device, history = train_completed_gan(data, args)
    selected_training_row = max(
        history, key=lambda row: float(row["selection_score"])
    )
    if args.stop_after_training:
        print(
            "VALIDATION-SELECTED ACCURACY-ONLY RESULT: "
            + json.dumps({
                key: selected_training_row[key]
                for key in (
                    "epoch",
                    "phase",
                    "generator_deterministic_position_accuracy_percent",
                    "generator_deterministic_velocity_accuracy_percent",
                    "generator_deterministic_full_state_accuracy_percent",
                    "generator_deterministic_mean_velocity_error",
                    "generator_deterministic_p95_velocity_error",
                    "minimum_deployable_validation_accuracy_percent",
                    "deployable_validation_target_reached",
                )
            }),
            flush=True,
        )
        print(
            "STOPPED AFTER TRAINING AS REQUESTED. The resumable checkpoint is: "
            f"{training_checkpoint_path(args)}",
            flush=True,
        )
        if bool(selected_training_row["deployable_validation_target_reached"]):
            print(
                "THE VALIDATION TARGET WAS REACHED. Final evaluation can be "
                "started later with the exact same arguments plus --resume "
                "and without --stop_after_training.",
                flush=True,
            )
        else:
            print(
                "THE VALIDATION TARGET WAS NOT REACHED. Do not start final "
                "discriminator training or rollout packaging yet; inspect this "
                "selected result before choosing the next accuracy experiment.",
                flush=True,
            )
        return
    if (
        not bool(selected_training_row["deployable_validation_target_reached"])
        and not args.allow_below_target_finalization
    ):
        print(
            "FINALIZATION BLOCKED: the validation-selected deployable generator "
            f"has minimum accuracy "
            f"{selected_training_row['minimum_deployable_validation_accuracy_percent']:.6f}% "
            f"and has not reached the requested {args.target_accuracy:.6f}%. "
            "No discriminator audit, rollout benchmark, pictures, or packaging "
            "was started.",
            flush=True,
        )
        print(
            "Continue accuracy work from the saved checkpoint, or explicitly "
            "add --allow_below_target_finalization if a below-target research "
            "baseline is intentionally required.",
            flush=True,
        )
        return
    # Testing labels are first summarized only after model training and
    # validation selection are irrevocably complete. These values cannot alter
    # weights, hyperparameters, or checkpoint choice in this run.
    print(
        "UNLOCKING HELD-OUT TESTING DIAGNOSTICS FOR FINAL EVALUATION",
        flush=True,
    )
    diagnostics = dataset_diagnostics(data, include_testing=True)
    target_diagnostics = target_condition_diagnostics(
        data, include_testing=True
    )
    ceiling = representation_ceiling(
        data,
        norm,
        state_threshold=args.state_accuracy_threshold,
        position_threshold=args.position_accuracy_threshold,
        velocity_threshold=args.velocity_accuracy_threshold,
        include_testing=True,
    )
    ambiguity = condition_ambiguity_diagnostics(
        data,
        norm,
        conflict_threshold=min(
            args.state_accuracy_threshold,
            args.position_accuracy_threshold,
            args.velocity_accuracy_threshold,
        ),
        include_testing=True,
    )
    C, final_d_report, final_d_log = train_final_discriminator(G, data, norm, device, args)
    create_package(
        data, G, D, C, norm, device, args, history, final_d_report,
        final_d_log, diagnostics, ceiling, ambiguity, target_diagnostics, split_overlap,
        benchmark_eligibility, condition_reconstruction,
    )


if __name__ == "__main__":
    main()
