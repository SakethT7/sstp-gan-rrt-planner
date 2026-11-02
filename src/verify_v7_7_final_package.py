#!/usr/bin/env python3
"""Independent read-only integrity audit for the final V7.7 folder and ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import zipfile
from collections import Counter
from pathlib import Path


def sha256_stream(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return sha256_stream(handle)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value):
    return int(float(value))


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


def require_file(path, failures):
    path = Path(path)
    if not path.is_file():
        failures.append(f"missing required file: {path}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    package = Path(args.package_dir).expanduser().resolve()
    archive = Path(str(package) + ".zip")
    failures = []
    warnings = []

    if not package.is_dir():
        raise FileNotFoundError(f"Final package folder not found: {package}")
    if not archive.is_file():
        raise FileNotFoundError(f"Final package ZIP not found: {archive}")

    required = [
        package / "FINAL_METRICS.json",
        package / "METRIC_DEFINITIONS.json",
        package / "verification_report.json",
        package / "FILE_MANIFEST_SHA256.json",
        package / "BUILD_COMPLETED_SUCCESSFULLY.txt",
        package / "04_OVERALL_BENCHMARK" / "02_OVERALL_BENCHMARK_DATA" / "all_rollout_cases.csv",
        package / "04_OVERALL_BENCHMARK" / "02_OVERALL_BENCHMARK_DATA" / "testing_classical_basic_rrt_cases.csv",
        package / "05_MODEL_AND_DATASET_FILES" / "model" / "v7_7_invariant_constraint_rrt_sample_gan_model.pt",
        package / "05_MODEL_AND_DATASET_FILES" / "model" / "V7_7_FINAL_TRAINING_CHECKPOINT.pt",
        package / "05_MODEL_AND_DATASET_FILES" / "model" / "VALIDATION_SELECTED_CHECKPOINT_SUMMARY.json",
        package / "07_CODE_AND_REPRODUCIBILITY" / "SOURCE_SUITE_INVENTORY.json",
        package / "08_RESULTS_GALLERY" / "GALLERY_VERIFICATION.json",
        package / "08_RESULTS_GALLERY" / "FIGURE_INDEX.csv",
        package / "08_RESULTS_GALLERY" / "PATH_FIGURE_INDEX.csv",
    ]
    for path in required:
        require_file(path, failures)
    if failures:
        raise RuntimeError("\n".join(failures))

    metrics = read_json(package / "FINAL_METRICS.json")
    verification = read_json(package / "verification_report.json")
    run_config = read_json(
        package / "05_MODEL_AND_DATASET_FILES" / "run_config.json"
    )
    gallery = read_json(
        package / "08_RESULTS_GALLERY" / "GALLERY_VERIFICATION.json"
    )
    source_inventory = read_json(
        package / "07_CODE_AND_REPRODUCIBILITY" / "SOURCE_SUITE_INVENTORY.json"
    )

    if not verification.get("package_structure_valid"):
        failures.append("trainer package_structure_valid is false")
    if int(verification.get("accepted_edge_collision_violations_across_benchmark", -1)) != 0:
        failures.append("verification report contains accepted collision violations")
    if not verification.get("runtime_rrt_contract_passed"):
        failures.append("verification report says the runtime RRT contract failed")
    if not gallery.get("every_figure_has_data"):
        failures.append("gallery figure/data one-to-one check failed")
    if gallery.get("fabricated_values") is not False:
        failures.append("gallery did not explicitly certify fabricated_values=false")
    if int(gallery.get("figures", 0)) < 25:
        failures.append("fewer than 25 numbered data figures were generated")
    if int(gallery.get("path_images", 0)) != 45:
        failures.append(
            f"expected 45 real path images, found {gallery.get('path_images')}"
        )
    if not source_inventory.get("all_requested_sources_copied"):
        failures.append("complete source suite was not preserved")
    if source_inventory.get("errors"):
        failures.append(f"source-copy errors: {source_inventory['errors']}")

    for split in ("training", "validation", "testing"):
        if split not in metrics:
            failures.append(f"FINAL_METRICS is missing split {split}")
            continue
        for key in (
            "generator_deterministic_position_accuracy_percent",
            "generator_deterministic_velocity_accuracy_percent",
            "generator_deterministic_full_state_accuracy_percent",
            "generator_deterministic_mean_velocity_error",
            "generator_deterministic_p95_velocity_error",
        ):
            value = metrics[split].get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(f"nonfinite or missing metric: {split}.{key}")

    benchmark_data = package / "04_OVERALL_BENCHMARK" / "02_OVERALL_BENCHMARK_DATA"
    rollout_rows = read_csv(benchmark_data / "all_rollout_cases.csv")
    cases_per_difficulty = int(run_config["benchmark_cases_per_difficulty"])
    expected_learned_rows = 3 * 3 * cases_per_difficulty
    if len(rollout_rows) != expected_learned_rows:
        failures.append(
            f"expected {expected_learned_rows} learned rollout rows, found {len(rollout_rows)}"
        )
    learned_counts = Counter(
        (row.get("split"), row.get("difficulty_bucket")) for row in rollout_rows
    )
    for split in ("training", "validation", "testing"):
        for difficulty in ("easy", "medium", "hard"):
            if learned_counts[(split, difficulty)] != cases_per_difficulty:
                failures.append(
                    f"wrong learned count for {split}/{difficulty}: "
                    f"{learned_counts[(split, difficulty)]}"
                )
    for row_number, row in enumerate(rollout_rows, start=2):
        if not as_bool(row.get("runtime_contract_passed")):
            failures.append(f"runtime contract failed in all_rollout_cases.csv row {row_number}")
        if as_int(row.get("accepted_edge_collision_violations", -1)) != 0:
            failures.append(f"accepted collision violation in rollout row {row_number}")
        if as_int(row.get("sample_condition_mismatch_count", -1)) != 0:
            failures.append(f"sample-condition mismatch in rollout row {row_number}")
        if as_int(row.get("max_children_added_in_any_iteration", 99)) > 1:
            failures.append(f"multiple children added in one iteration at row {row_number}")
        if row.get("scenario_eligibility") != "valid_nontrivial":
            failures.append(f"ineligible scenario entered rollout row {row_number}")

    classical_rows = read_csv(
        benchmark_data / "testing_classical_basic_rrt_cases.csv"
    )
    expected_classical_rows = 3 * cases_per_difficulty
    if len(classical_rows) != expected_classical_rows:
        failures.append(
            f"expected {expected_classical_rows} classical rows, found {len(classical_rows)}"
        )
    for row_number, row in enumerate(classical_rows, start=2):
        if not as_bool(row.get("runtime_contract_passed")):
            failures.append(f"classical runtime contract failed at row {row_number}")
        if as_int(row.get("accepted_edge_collision_violations", -1)) != 0:
            failures.append(f"classical accepted collision violation at row {row_number}")

    figure_rows = read_csv(package / "08_RESULTS_GALLERY" / "FIGURE_INDEX.csv")
    if len(figure_rows) != int(gallery.get("figures", -1)):
        failures.append("FIGURE_INDEX count disagrees with gallery verification")
    for row in figure_rows:
        for key in ("png", "svg", "data_csv"):
            if not (package / "08_RESULTS_GALLERY" / row[key]).is_file():
                failures.append(f"figure index points to missing {key}: {row[key]}")

    path_rows = read_csv(
        package / "08_RESULTS_GALLERY" / "PATH_FIGURE_INDEX.csv"
    )
    if len(path_rows) != 45:
        failures.append(f"PATH_FIGURE_INDEX has {len(path_rows)} rows instead of 45")
    for row in path_rows:
        image = row.get("image", "")
        if not (
            package / "08_RESULTS_GALLERY" / "04_PATH_IMAGES" / image
        ).is_file():
            failures.append(f"path index points to missing image: {image}")

    manifest_path = package / "FILE_MANIFEST_SHA256.json"
    manifest = read_json(manifest_path)
    disk_files = {
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file() and path != manifest_path
    }
    manifest_files = set(manifest)
    if disk_files != manifest_files:
        for missing in sorted(disk_files - manifest_files):
            failures.append(f"file absent from manifest: {missing}")
        for missing in sorted(manifest_files - disk_files):
            failures.append(f"manifest entry absent from disk: {missing}")
    hash_started = time.perf_counter()
    for index, relative in enumerate(sorted(manifest), start=1):
        record = manifest[relative]
        path = package / relative
        if not path.is_file():
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            failures.append(f"disk SHA-256 mismatch: {relative}")
        if int(path.stat().st_size) != int(record.get("size", -1)):
            failures.append(f"disk size mismatch: {relative}")
        if index % 100 == 0 or index == len(manifest):
            elapsed = time.perf_counter() - hash_started
            eta = elapsed / index * max(0, len(manifest) - index)
            print(
                f"DISK HASH AUDIT: {index}/{len(manifest)} | "
                f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}",
                flush=True,
            )

    zip_started = time.perf_counter()
    prefix = package.name + "/"
    with zipfile.ZipFile(archive, "r") as zipped:
        zip_names = {
            name for name in zipped.namelist() if not name.endswith("/")
        }
        expected_zip_names = {
            prefix + relative for relative in manifest
        } | {prefix + "FILE_MANIFEST_SHA256.json"}
        if zip_names != expected_zip_names:
            for missing in sorted(expected_zip_names - zip_names):
                failures.append(f"ZIP is missing: {missing}")
            for extra in sorted(zip_names - expected_zip_names):
                failures.append(f"ZIP contains unmanifested file: {extra}")
        zip_targets = sorted(manifest)
        for index, relative in enumerate(zip_targets, start=1):
            member = prefix + relative
            if member not in zip_names:
                continue
            with zipped.open(member, "r") as handle:
                actual_hash = sha256_stream(handle)
            if actual_hash != manifest[relative].get("sha256"):
                failures.append(f"ZIP SHA-256 mismatch: {relative}")
            if index % 100 == 0 or index == len(zip_targets):
                elapsed = time.perf_counter() - zip_started
                eta = elapsed / index * max(0, len(zip_targets) - index)
                print(
                    f"ZIP HASH AUDIT: {index}/{len(zip_targets)} | "
                    f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}",
                    flush=True,
                )

    result = {
        "status": "passed" if not failures else "failed",
        "package": str(package),
        "zip": str(archive),
        "trainer_verification_status": verification.get("status"),
        "all_requested_targets_reached": verification.get(
            "all_requested_targets_reached"
        ),
        "learned_rollout_rows": len(rollout_rows),
        "classical_rollout_rows": len(classical_rows),
        "numbered_data_figures": len(figure_rows),
        "real_path_images": len(path_rows),
        "manifested_files": len(manifest),
        "failures": failures,
        "warnings": warnings,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit("FINAL PACKAGE INTEGRITY AUDIT FAILED")
    print("FINAL PACKAGE INTEGRITY AUDIT PASSED", flush=True)


if __name__ == "__main__":
    main()
