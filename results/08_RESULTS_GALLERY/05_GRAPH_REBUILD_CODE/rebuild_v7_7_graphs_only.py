#!/usr/bin/env python3
"""Transactionally replace only V7.7 graph products and their indexes.

Authoritative model, dataset, checkpoint, raw metric JSON, and rollout CSV files
are hashed before and after the rebuild. Old graph directories are moved into a
temporary rollback area and deleted only after the new gallery, mirrors,
manifest, ZIP, and independent integrity audit all pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


GRAPH_DIRECTORIES = (
    "08_RESULTS_GALLERY",
    "04_OVERALL_BENCHMARK/01_OVERALL_BENCHMARK_CHARTS",
    "05_MODEL_AND_DATASET_FILES/metrics_master/FINAL_POLISHED_CHARTS",
    "05_MODEL_AND_DATASET_FILES/metrics_master/TRAINING_PROGRESSION_CHARTS",
    "05_MODEL_AND_DATASET_FILES/metrics_master/FINAL_DISCRIMINATOR_PROGRESSION_CHARTS",
)
MANIFEST_NAME = "FILE_MANIFEST_SHA256.json"

METRIC_SLUGS = (
    "deterministic_accuracy_by_split",
    "velocity_accuracy_thresholds",
    "position_accuracy_thresholds",
    "full_state_accuracy_thresholds",
    "velocity_error_by_split",
    "position_error_by_split",
    "full_state_error_by_split",
    "full_state_mse_by_split",
    "deterministic_vs_oracle_full_state_accuracy",
    "deterministic_vs_oracle_full_state_mse",
    "control_increment_accuracy",
    "control_increment_error",
    "deterministic_vs_oracle_velocity",
    "candidate_diversity",
)
TRAINING_SLUGS = (
    "validation_accuracy_progression",
    "training_validation_velocity_accuracy",
    "validation_velocity_error_progression",
    "loss_progression",
    "learning_rate_schedule",
    "generalization_gap",
    "validation_velocity_threshold_progression",
    "validation_control_increment_accuracy_progression",
)
DISCRIMINATOR_SLUGS = (
    "final_discriminator_accuracy",
    "final_discriminator_loss",
)
ROLLOUT_SLUGS = (
    "rollout_solve_rate",
    "rollout_path_cost",
    "rollout_runtime",
    "rollout_tree_nodes",
    "rollout_goal_distance",
    "gan_node_share",
    "paired_solve_rate",
    "rollout_iterations",
    "planner_rejections",
    "planner_effort",
    "generated_and_accepted_nodes",
    "paired_runtime",
    "paired_path_length",
    "paired_outcomes",
    "accepted_node_source_composition",
    "observed_failure_reasons",
    "testing_runtime_distribution",
    "testing_path_length_distribution",
    "testing_effort_cdf",
    "paired_case_path_scatter",
    "paired_case_runtime_scatter",
    "overall_solve_rate_cluster_intervals",
)

REQUIRED_ROLLOUT_NAMES = {
    "rollout_solve_rate": "01_rollout_solve_rate",
    "rollout_path_cost": "02_rollout_solved_path_length",
    "rollout_runtime": "03_rollout_runtime",
    "rollout_tree_nodes": "04_rollout_tree_nodes",
    "rollout_goal_distance": "05_rollout_goal_distance",
    "gan_node_share": "06_gan_share_of_accepted_nodes",
    "paired_solve_rate": "07_testing_gan_vs_classical_solve_rate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def under_graph_directory(relative: Path) -> bool:
    value = relative.as_posix()
    return any(
        value == prefix or value.startswith(prefix + "/")
        for prefix in GRAPH_DIRECTORIES
    )


def protected_hashes(package: Path):
    values = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package)
        if relative.as_posix() == MANIFEST_NAME or under_graph_directory(relative):
            continue
        values[relative.as_posix()] = sha256_file(path)
    return values


def directory_hashes(root: Path):
    """Hash every file in a directory using paths relative to that directory."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v7_7_gallery_rebuild_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import graph builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_entry(entries, slug):
    suffixes = (f"_{slug}.png", f"_{slug}.svg", f"_{slug}.csv")
    for entry in entries:
        if str(entry.get("png", "")).endswith(suffixes[0]):
            return entry
    return None


def build_mirror(package: Path, gallery: Path, relative_target: str, slugs):
    target = package / relative_target
    target.mkdir(parents=True, exist_ok=True)
    entries = read_csv(gallery / "FIGURE_INDEX.csv")
    index_rows = []
    sequence = 1
    for slug in slugs:
        entry = find_entry(entries, slug)
        if entry is None:
            continue
        if relative_target.endswith("01_OVERALL_BENCHMARK_CHARTS") and slug in REQUIRED_ROLLOUT_NAMES:
            stem = REQUIRED_ROLLOUT_NAMES[slug]
        else:
            stem = f"{sequence:02d}_{slug}"
        source_png = gallery / entry["png"]
        source_svg = gallery / entry["svg"]
        source_csv = gallery / entry["data_csv"]
        destination_png = target / f"{stem}.png"
        destination_svg = target / f"{stem}.svg"
        destination_csv = target / f"{stem}.csv"
        for source, destination in (
            (source_png, destination_png),
            (source_svg, destination_svg),
            (source_csv, destination_csv),
        ):
            if not source.is_file():
                raise FileNotFoundError(f"Missing gallery product: {source}")
            shutil.copy2(source, destination)
        index_rows.append({
            "order": sequence,
            "title": entry["title"],
            "caption": entry["caption"],
            "png": destination_png.name,
            "svg": destination_svg.name,
            "data_csv": destination_csv.name,
        })
        sequence += 1
    if not index_rows:
        raise RuntimeError(f"No replacement graphs were copied to {target}")
    write_csv(target / "GRAPH_INDEX.csv", index_rows)
    (target / "README.txt").write_text(
        "Replacement graph set generated only from authoritative final JSON/CSV data.\n"
        "Percentages display two decimals. Ordinary values display at most two "
        "decimals; small losses/errors use scientific notation. Matching CSV "
        "files retain full source precision.\n",
        encoding="utf-8",
    )


def validate_gallery(gallery: Path):
    verification_path = gallery / "GALLERY_VERIFICATION.json"
    if not verification_path.is_file():
        raise RuntimeError("Replacement gallery did not produce verification metadata")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    figures = int(verification.get("figures", 0))
    if figures < 40:
        raise RuntimeError(f"Expected at least 40 useful figures, found {figures}")
    if int(verification.get("path_images", 0)) != 45:
        raise RuntimeError(
            f"Expected 45 preserved real path images, found {verification.get('path_images')}"
        )
    if not verification.get("every_figure_has_data"):
        raise RuntimeError("At least one replacement figure lacks matching CSV data")
    if verification.get("fabricated_values") is not False:
        raise RuntimeError("Replacement gallery did not certify fabricated_values=false")
    policy = verification.get("display_precision_policy", {})
    if int(policy.get("percentages_decimal_places", -1)) != 2:
        raise RuntimeError("Replacement gallery does not enforce two-decimal percentages")
    if policy.get("source_csv_json_precision_modified") is not False:
        raise RuntimeError("Replacement gallery claims source precision was modified")


def copy_graph_rebuild_provenance(gallery: Path, builder: Path, rebuilder: Path):
    """Store the exact graph-only source inside the graph product it generated."""
    target = gallery / "05_GRAPH_REBUILD_CODE"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in (builder, rebuilder):
        destination = target / source.name
        shutil.copy2(source, destination)
        copied.append({
            "file": destination.name,
            "sha256": sha256_file(destination),
        })
    (target / "GRAPH_CODE_MANIFEST.json").write_text(
        json.dumps({
            "purpose": "Exact source used for this graph-only replacement",
            "files": copied,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "README.txt").write_text(
        "These are the exact scripts used to rebuild only the graph products.\n"
        "Model weights, datasets, checkpoints, raw metrics, rollout rows, and "
        "path images were protected by before/after SHA-256 comparison.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Safely replace only V7.7 graph products with compact-decimal expanded figures."
    )
    parser.add_argument(
        "--package_dir", default=str(Path.home() / "V7_7_GENERALIZATION_GAN_FINAL")
    )
    args = parser.parse_args()
    package = Path(args.package_dir).expanduser().resolve()
    source_dir = Path(__file__).resolve().parent
    builder = source_dir / "build_v7_7_results_gallery.py"
    rebuilder = Path(__file__).resolve()
    verifier = source_dir / "verify_v7_7_final_package.py"
    archive = Path(str(package) + ".zip")

    if package in {Path("/").resolve(), Path.home().resolve()} or package.is_symlink():
        raise ValueError(f"Unsafe package directory: {package}")
    required = (
        package / "FINAL_METRICS.json",
        package / "verification_report.json",
        package / "04_OVERALL_BENCHMARK/02_OVERALL_BENCHMARK_DATA/all_rollout_cases.csv",
        package / "08_RESULTS_GALLERY/GALLERY_VERIFICATION.json",
        package / "08_RESULTS_GALLERY/04_PATH_IMAGES/rendered_case_index.csv",
        archive,
        builder,
        verifier,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Final pipeline or graph update is incomplete; missing: " + "; ".join(missing)
        )
    for relative in GRAPH_DIRECTORIES:
        target = package / relative
        if target.is_symlink():
            raise ValueError(f"Graph target must not be a symlink: {target}")

    print("GRAPH-ONLY REBUILD PREFLIGHT", flush=True)
    print("Protected model/data/checkpoint/raw-metric files are being hashed...", flush=True)
    before = protected_hashes(package)
    path_image_before = directory_hashes(
        package / "08_RESULTS_GALLERY/04_PATH_IMAGES"
    )
    if len([name for name in path_image_before if name.lower().endswith(".png")]) != 45:
        raise RuntimeError("Expected exactly 45 existing path PNGs before rebuilding graphs")
    subprocess.run([sys.executable, str(builder), "--self_test"], check=True)
    # Refuse to modify an incomplete or already-corrupt final package.
    subprocess.run([
        sys.executable, str(verifier), "--package_dir", str(package)
    ], check=True)

    rollback_root = Path(tempfile.mkdtemp(
        prefix=f".{package.name}_graph_rollback_", dir=package.parent
    ))
    moved_directories = []
    manifest = package / MANIFEST_NAME
    backup_manifest = rollback_root / MANIFEST_NAME
    backup_archive = rollback_root / archive.name
    try:
        for relative in GRAPH_DIRECTORIES:
            source = package / relative
            if source.exists():
                destination = rollback_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved_directories.append(relative)
        if manifest.is_file():
            shutil.move(str(manifest), str(backup_manifest))
        if archive.is_file():
            shutil.move(str(archive), str(backup_archive))

        preserved_paths = rollback_root / "08_RESULTS_GALLERY/04_PATH_IMAGES"
        if len(list(preserved_paths.rglob("*.png"))) != 45:
            raise RuntimeError("Rollback copy does not contain exactly 45 real path images")

        subprocess.run([
            sys.executable, str(builder),
            "--package_dir", str(package),
            "--path_images_dir", str(preserved_paths),
            "--overwrite",
        ], check=True)
        gallery = package / "08_RESULTS_GALLERY"
        validate_gallery(gallery)
        path_image_after = directory_hashes(gallery / "04_PATH_IMAGES")
        if path_image_after != path_image_before:
            raise RuntimeError("A protected real path image or its index changed")
        copy_graph_rebuild_provenance(gallery, builder, rebuilder)

        build_mirror(
            package, gallery,
            "05_MODEL_AND_DATASET_FILES/metrics_master/FINAL_POLISHED_CHARTS",
            METRIC_SLUGS,
        )
        build_mirror(
            package, gallery,
            "05_MODEL_AND_DATASET_FILES/metrics_master/TRAINING_PROGRESSION_CHARTS",
            TRAINING_SLUGS,
        )
        build_mirror(
            package, gallery,
            "05_MODEL_AND_DATASET_FILES/metrics_master/FINAL_DISCRIMINATOR_PROGRESSION_CHARTS",
            DISCRIMINATOR_SLUGS,
        )
        build_mirror(
            package, gallery,
            "04_OVERALL_BENCHMARK/01_OVERALL_BENCHMARK_CHARTS",
            ROLLOUT_SLUGS,
        )

        gallery_module = load_module(builder)
        gallery_module.refresh_manifest_and_zip(package)
        after = protected_hashes(package)
        if before != after:
            changed = sorted(set(before) ^ set(after))
            changed.extend(
                key for key in sorted(set(before) & set(after))
                if before[key] != after[key]
            )
            raise RuntimeError(
                "A protected non-graph file changed: " + "; ".join(changed[:20])
            )

        subprocess.run([
            sys.executable, str(verifier), "--package_dir", str(package)
        ], check=True)
        shutil.rmtree(rollback_root)
        print("GRAPH-ONLY REBUILD FINISHED SUCCESSFULLY", flush=True)
        print("Old graph folders were deleted after the replacement passed.", flush=True)
        print("Protected model, dataset, checkpoints, raw metrics, and rollout rows are unchanged.", flush=True)
        print(f"FINAL FOLDER: {package}", flush=True)
        print(f"FINAL ZIP: {archive}", flush=True)
    except Exception:
        print("GRAPH REBUILD FAILED; RESTORING ORIGINAL GRAPH PRODUCTS", flush=True)
        for relative in GRAPH_DIRECTORIES:
            new_target = package / relative
            if new_target.exists():
                shutil.rmtree(new_target)
        for relative in moved_directories:
            backup = rollback_root / relative
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                shutil.move(str(backup), str(destination))
        if manifest.exists():
            manifest.unlink()
        if backup_manifest.exists():
            shutil.move(str(backup_manifest), str(manifest))
        if archive.exists():
            archive.unlink()
        if backup_archive.exists():
            shutil.move(str(backup_archive), str(archive))
        shutil.rmtree(rollback_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
