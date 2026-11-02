#!/usr/bin/env python3
"""Render selected V7.7 rollout paths after training/finalization is complete."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_NAME = "build_v7_7_rrt_sample_gan.py"
EXPECTED_VERSION = "7.7.0-invariant-constraint-attention-control-gan"


def import_build_module():
    source = Path(__file__).resolve().with_name(SCRIPT_NAME)
    if not source.is_file():
        raise FileNotFoundError(
            f"Required trainer is not beside this renderer: {source}"
        )
    spec = importlib.util.spec_from_file_location("v7_7_build_module", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module specification: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.SCRIPT_VERSION != EXPECTED_VERSION:
        raise RuntimeError(
            f"Renderer expects {EXPECTED_VERSION!r}, found {module.SCRIPT_VERSION!r}"
        )
    return module


def load_torch_file(module, path, device):
    return module.torch_load_checkpoint(path, map_location=device)


def restore_normalizer(saved):
    vector_keys = {
        "x_mean", "x_std", "p_mean", "p_std", "local_p_mean", "local_p_std",
        "control_increment_mean", "control_increment_std",
        "local_control_increment_mean", "local_control_increment_std",
    }
    return {
        key: np.asarray(value, dtype=np.float32) if key in vector_keys else value
        for key, value in saved.items()
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evenly_trim(selected, total):
    groups = {
        bucket: [item for item in selected if item[1] == bucket]
        for bucket in ("easy", "medium", "hard")
    }
    answer = []
    while len(answer) < total:
        added = False
        for bucket in ("easy", "medium", "hard"):
            if groups[bucket] and len(answer) < total:
                answer.append(groups[bucket].pop(0))
                added = True
        if not added:
            break
    return answer


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


def direct_route_is_blocked(module, condition, goal_state, margin):
    start = np.asarray(condition[0:2], dtype=float)
    goal = np.asarray(goal_state[0:2], dtype=float)
    scenario = np.asarray(condition, dtype=np.float32).copy()
    scenario[4:6] = goal
    for ox, oy, radius in module.parse_obstacles(scenario):
        clearance = module.point_segment_distance((ox, oy), start, goal)
        if clearance <= float(radius) + float(margin):
            return True
    return False


def main():
    render_started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Render real V7.7 GAN-RRT rollout paths from a completed package."
    )
    parser.add_argument("--package_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pictures_per_split", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args_cli = parser.parse_args()
    if args_cli.pictures_per_split < 1:
        raise ValueError("--pictures_per_split must be at least 1")

    package = Path(args_cli.package_dir).expanduser().resolve()
    if not package.is_dir():
        raise FileNotFoundError(f"Package directory was not found: {package}")
    output = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else Path(str(package) + "_PATH_PICTURES")
    )
    if output in {Path("/").resolve(), Path.home().resolve(), package}:
        raise ValueError(f"Unsafe picture output directory: {output}")
    if output.is_symlink():
        raise ValueError(f"Picture output directory must not be a symlink: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Picture output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if not args_cli.overwrite:
            raise FileExistsError(
                f"Picture output directory is not empty: {output}. "
                "Choose another directory or add --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    module = import_build_module()
    run_config_path = package / "05_MODEL_AND_DATASET_FILES" / "run_config.json"
    model_path = (
        package
        / "05_MODEL_AND_DATASET_FILES"
        / "model"
        / "v7_7_invariant_constraint_rrt_sample_gan_model.pt"
    )
    dataset_path = (
        package / "05_MODEL_AND_DATASET_FILES" / "rrt_sample_gan_dataset_v7.npz"
    )
    for required in (run_config_path, model_path, dataset_path):
        if not required.is_file():
            raise FileNotFoundError(f"Completed package is missing: {required}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    planner_args = SimpleNamespace(**run_config)
    planner_args.cpu = bool(args_cli.cpu)
    planner_args.max_turn_rad = math.radians(float(planner_args.max_turn_deg))
    data = module.load_dataset(dataset_path)
    planner_args.runtime_effective_dt_normalized = float(
        data["supervision_lookahead_steps"]
        * data["nominal_dt"]
        * data["velocity_scale"]
        / float(data["position_scale"][0])
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args_cli.cpu else "cpu"
    )
    checkpoint = load_torch_file(module, model_path, device)
    if checkpoint.get("script_version") != EXPECTED_VERSION:
        raise RuntimeError("Model package belongs to a different trainer version")
    norm = restore_normalizer(checkpoint["normalizer"])
    noise_dim = int(checkpoint["noise_dim"])
    condition_dim = int(data["train_X"].shape[1])
    generator = module.LocalMotionGenerator(condition_dim, norm, noise_dim).to(device)
    discriminator = module.LocalMotionDiscriminator(condition_dim).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(
        checkpoint["adversarial_discriminator_state_dict"]
    )
    generator.eval()
    discriminator.eval()

    rows = []
    split_specs = [
        ("training", "train_X", "training_goal_state"),
        ("validation", "validation_X", "validation_goal_state"),
        ("testing", "testing_X", "testing_goal_state"),
    ]
    per_bucket = int(math.ceil(args_cli.pictures_per_split / 3.0))
    total_requested = args_cli.pictures_per_split * len(split_specs)
    completed = 0
    for split_index, (split, xkey, goal_key) in enumerate(split_specs):
        candidate_pool_per_bucket = max(30, per_bucket * 8)
        candidate_pool, _ = module.selected_benchmark_cases(
            data[xkey],
            data[goal_key],
            candidate_pool_per_bucket,
            args_cli.seed + 10007 * split_index,
            planner_args,
        )
        selected = []
        for bucket in ("easy", "medium", "hard"):
            bucket_candidates = [
                item for item in candidate_pool if item[1] == bucket
            ]
            bucket_candidates.sort(
                key=lambda item: (
                    not direct_route_is_blocked(
                        module,
                        data[xkey][item[0]],
                        data[goal_key][item[0]],
                        planner_args.collision_margin,
                    ),
                    -float(item[2]),
                    int(item[0]),
                )
            )
            selected.extend(bucket_candidates[:per_bucket])
        selected = evenly_trim(selected, args_cli.pictures_per_split)
        for case_number, (row_index, bucket, difficulty) in enumerate(
            selected, start=1
        ):
            result = module.gan_rrt_plan(
                generator,
                discriminator,
                data[xkey][row_index].astype(np.float32),
                data[goal_key][row_index].astype(np.float32),
                norm,
                device,
                planner_args,
                seed=args_cli.seed + 1000003 * split_index + row_index,
            )
            split_dir = output / split.upper() / bucket.upper()
            split_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{case_number:03d}_{split}_{bucket}_row_{row_index:05d}_"
                f"{'SOLVED' if result['solved'] else 'FAILED'}.png"
            )
            picture = split_dir / filename
            module.plot_gan_rrt(
                result,
                picture,
                title=(
                    f"V7.7 GAN-RRT — {split.title()} / "
                    f"{bucket.title()} / row {row_index}"
                ),
            )
            rows.append(
                {
                    "split": split,
                    "difficulty": bucket,
                    "row_index": int(row_index),
                    "difficulty_score": float(difficulty),
                    "direct_start_to_goal_route_blocked": bool(
                        direct_route_is_blocked(
                            module,
                            data[xkey][row_index],
                            data[goal_key][row_index],
                            planner_args.collision_margin,
                        )
                    ),
                    "solved": bool(result["solved"]),
                    "reason": result["reason"],
                    "nodes": len(result["nodes"]),
                    "iterations": int(result["iterations"]),
                    "path_cost": float(result["path_cost"]),
                    "runtime_contract_passed": bool(
                        result["runtime_contract_passed"]
                    ),
                    "accepted_edge_collision_violations": int(
                        result["accepted_edge_collision_violations"]
                    ),
                    "picture": str(picture.relative_to(output)),
                }
            )
            completed += 1
            elapsed = time.perf_counter() - render_started
            eta = elapsed / completed * max(0, total_requested - completed)
            print(
                f"PATH IMAGE PROGRESS: {completed}/{total_requested} "
                f"({100.0 * completed / max(1, total_requested):.2f}%) | "
                f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)} | "
                f"wrote={picture}",
                flush=True,
            )
    write_csv(output / "rendered_case_index.csv", rows)
    if not rows or not all(row["runtime_contract_passed"] for row in rows):
        raise RuntimeError("At least one rendered rollout failed the RRT contract")
    if any(row["accepted_edge_collision_violations"] for row in rows):
        raise RuntimeError("At least one rendered rollout accepted an invalid edge")
    blocked_count = sum(
        bool(row["direct_start_to_goal_route_blocked"]) for row in rows
    )
    print(
        f"DONE: wrote {len(rows)} real rollout pictures to {output} | "
        f"direct-route-blocked cases={blocked_count}/{len(rows)} | "
        f"elapsed={format_duration(time.perf_counter() - render_started)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
