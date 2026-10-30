"""
==============================================================================
SOALM V2 — COMPLETE MODEL & DATASET EVALUATOR
==============================================================================

Purpose
-------
Inspect the current SOALM V2 system without training anything.

Reports:

    ✓ Dataset size
    ✓ Dataset columns
    ✓ Feature count
    ✓ Regression targets
    ✓ Classification targets
    ✓ Trained .joblib models
    ✓ Missing models
    ✓ Checkpoints
    ✓ Regression validation MAE
    ✓ Classification validation accuracy
    ✓ Classification class distribution
    ✓ Training sample size
    ✓ Model storage
    ✓ Dataset storage
    ✓ Overall SOALM V2 readiness
    ✓ Detailed target-by-target status

This script DOES NOT:
    ✗ Train models
    ✗ Modify models
    ✗ Modify datasets
    ✗ Delete files
    ✗ Download anything
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pyarrow.parquet as pq

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERA5_ROOT = PROJECT_ROOT / "data" / "era5"

# Keep these paths aligned with scripts/train_era5_soalm_v2.py.
TRAINING_FILE = ERA5_ROOT / "prepared" / "soalm_v2_5m.parquet"

MODEL_ROOT = ERA5_ROOT / "models_v2_final"

CHECKPOINT_ROOT = ERA5_ROOT / "checkpoints_v2_final"

LOG_ROOT = ERA5_ROOT / "logs_v2_final"

FEATURE_METADATA_FILE = ERA5_ROOT / "features_v2_final" / "soalm_v2_features.json"

TRAINING_SUMMARY_FILE = LOG_ROOT / "training_summary.json"


# =============================================================================
# SOALM V2 FEATURES
# =============================================================================

SOALM_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "showers",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "rain_last_1h",
    "rain_last_3h",
    "rain_last_6h",
    "rain_last_12h",
    "rain_last_24h",
    "rain_last_48h",
    "rain_last_72h",
    "pressure_change_1h",
    "pressure_change_3h",
    "pressure_change_6h",
    "temperature_change_3h",
    "temperature_change_6h",
    "humidity_change_3h",
    "humidity_change_6h",
    "wind_change_3h",
    "wind_change_6h",
    "latitude",
    "longitude",
    "elevation",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# =============================================================================
# TARGETS
# =============================================================================

REGRESSION_TARGETS = [
    "rain_future_1h",
    "rain_future_3h",
    "rain_future_6h",
    "rain_future_12h",
    "rain_future_24h",
    "rain_future_48h",
    "rain_future_72h",
    "rain_accum_future_3h",
    "rain_accum_future_6h",
    "rain_accum_future_12h",
    "rain_accum_future_24h",
    "rain_accum_future_48h",
    "rain_accum_future_72h",
    "rain_max_future_3h",
    "rain_max_future_6h",
    "rain_max_future_12h",
    "rain_max_future_24h",
    "rain_max_future_48h",
    "rain_max_future_72h",
]


CLASSIFICATION_TARGETS = [
    "risk_1h",
    "risk_3h",
    "risk_6h",
    "risk_12h",
    "risk_24h",
    "heavy_rain_event_24h",
    "extreme_rain_event_24h",
    "dangerous_accumulation_24h",
    "critical_accumulation_24h",
    "rain_intensification_6h",
    "rain_intensification_24h",
]


ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS


# =============================================================================
# COLORS / TERMINAL HELPERS
# =============================================================================

USE_COLOR = sys.stdout.isatty()


def color(text, code):

    if not USE_COLOR:
        return text

    return f"\033[{code}m{text}\033[0m"


def green(text):
    return color(text, "92")


def yellow(text):
    return color(text, "93")


def red(text):
    return color(text, "91")


def cyan(text):
    return color(text, "96")


def bold(text):
    return color(text, "1")


def success(text):
    print(green("[OK] " + text))


def warning(text):
    print(yellow("[WARN] " + text))


def failure(text):
    print(red("[ERROR] " + text))


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def separator():

    print("=" * 78)


def human_size(size_bytes):

    size = float(size_bytes)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    for unit in units:

        if size < 1024:

            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def percentage(value):

    return f"{value:.2%}" if isinstance(value, (int, float)) else "N/A"


def directory_size(path):

    if not path.exists():

        return 0

    total = 0

    for root, dirs, files in os.walk(path):

        for file in files:

            try:

                total += (Path(root) / file).stat().st_size

            except OSError:

                pass

    return total


def file_size(path):

    try:

        return path.stat().st_size

    except OSError:

        return 0


def model_path(target):

    return MODEL_ROOT / f"{target}.joblib"


def metadata_path(target):

    # The current trainer stores validation metrics and class information in
    # per-target checkpoint JSON files rather than separate metadata files.
    return checkpoint_path(target)


def checkpoint_path(target):

    return CHECKPOINT_ROOT / f"{target}.json"


def load_json(path):

    if not path.exists():

        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return None


def configure_console():

    # Windows terminals using cp1252 cannot encode symbols such as ✓ and ✗.
    # Replacing unsupported characters keeps the diagnostic report running.
    for stream in (sys.stdout, sys.stderr):

        if hasattr(stream, "reconfigure"):

            try:

                stream.reconfigure(errors="replace")

            except (OSError, ValueError):

                pass

# =============================================================================
# HEADER
# =============================================================================


def print_header():

    print()

    separator()

    print(bold("SOALM V2 - COMPLETE SYSTEM STATUS"))

    separator()

    print()

    print(f"Evaluation time: " f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print()

    print(f"Project root:\n  {PROJECT_ROOT}")

    print(f"\nERA5 root:\n  {ERA5_ROOT}")


# =============================================================================
# DATASET STATUS
# =============================================================================


def evaluate_dataset():

    print()

    separator()

    print(bold("1. TRAINING DATASET"))

    separator()

    if not TRAINING_FILE.exists():

        failure("SOALM V2 training dataset NOT FOUND")

        print(f"\nExpected:\n  {TRAINING_FILE}")

        return {
            "exists": False,
            "rows": 0,
            "columns": 0,
        }

    size = file_size(TRAINING_FILE)

    print()
    print(f"File:")
    print(f"  {TRAINING_FILE}")

    print()
    print(f"Storage size: {human_size(size)}")

    print()
    print("Reading Parquet metadata...")

    try:

        parquet_file = pq.ParquetFile(TRAINING_FILE)
        rows = parquet_file.metadata.num_rows
        dataset_columns = set(parquet_file.schema.names)
        columns = len(dataset_columns)

        print()
        print(f"Rows:       {rows:,}")

        print(f"Columns:    {columns:,}")

        print()

        missing_features = [
            feature for feature in SOALM_FEATURES if feature not in dataset_columns
        ]

        missing_targets = [
            target for target in ALL_TARGETS if target not in dataset_columns
        ]

        if not missing_features:

            success(f"All {len(SOALM_FEATURES)} SOALM features present")

        else:

            failure(f"{len(missing_features)} features missing")

            for item in missing_features:

                print(f"    - {item}")

        if not missing_targets:

            success(f"All {len(ALL_TARGETS)} targets present")

        else:

            failure(f"{len(missing_targets)} targets missing")

            for item in missing_targets:

                print(f"    - {item}")

        # ---------------------------------------------------------------------
        # Count target values in batches so a multi-million-row parquet file
        # does not need to be loaded into memory all at once.
        target_counts = {target: 0 for target in ALL_TARGETS if target in dataset_columns}

        if target_counts:

            for batch in parquet_file.iter_batches(
                columns=list(target_counts),
                batch_size=100_000,
            ):

                for target, column in zip(batch.schema.names, batch.columns):

                    target_counts[target] += len(column) - column.null_count

        # ---------------------------------------------------------------------
        # Target availability
        # ---------------------------------------------------------------------

        print()

        print(bold("Target data availability:"))

        for target in ALL_TARGETS:

            if target not in dataset_columns:

                print(f"  {target:<40} MISSING")

                continue

            non_null = target_counts[target]

            percentage = non_null / rows * 100 if rows else 0

            print(f"  {target:<40} " f"{non_null:,} " f"({percentage:.2f}%)")

        return {
            "exists": True,
            "rows": rows,
            "columns": columns,
            "size": size,
            "missing_features": missing_features,
            "missing_targets": missing_targets,
        }

    except Exception as exc:

        failure(f"Could not inspect dataset: {exc}")

        return {
            "exists": True,
            "rows": 0,
            "columns": 0,
            "size": size,
        }


# =============================================================================
# FEATURE METADATA
# =============================================================================


def evaluate_feature_metadata():

    print()

    separator()

    print(bold("2. FEATURE METADATA"))

    separator()

    if not FEATURE_METADATA_FILE.exists():

        warning("Feature metadata file not found")

        return None

    metadata = load_json(FEATURE_METADATA_FILE)

    if not metadata:

        failure("Could not read feature metadata")

        return None

    print()

    print(f"Feature count: " f"{metadata.get('feature_count', 'N/A')}")

    features = metadata.get(
        "features",
        [],
    )

    print(f"Stored features: " f"{len(features)}")

    if features:

        print()

        print("Features:")

        for index, feature in enumerate(
            features,
            1,
        ):

            print(f"  {index:02d}. {feature}")

    medians = metadata.get(
        "feature_medians",
        {},
    )

    print()

    print(f"Stored feature medians: " f"{len(medians)}")

    success("Feature preprocessing metadata available")

    return metadata


# =============================================================================
# REGRESSION MODELS
# =============================================================================


def evaluate_regression_models():

    print()

    separator()

    print(bold("3. REGRESSION MODELS"))

    separator()

    trained = []

    missing = []

    total_size = 0

    print()

    print(f"Expected regression models: " f"{len(REGRESSION_TARGETS)}")

    print()

    print(f"{'STATUS':<10} " f"{'TARGET':<38} " f"{'MAE':<12} " f"{'SIZE':<12}")

    print("-" * 78)

    for target in REGRESSION_TARGETS:

        path = model_path(target)

        metadata = load_json(metadata_path(target))

        if path.exists():

            size = file_size(path)

            total_size += size

            trained.append(target)

            metrics = metadata.get("metrics", {}) if metadata else {}
            mae = metrics.get("mae") if metrics else None

            mae_text = f"{mae:.4f}" if isinstance(mae, (int, float)) else "N/A"

            print(
                f"{green('READY'):<19}"
                f"{target:<38}"
                f"{mae_text:<12}"
                f"{human_size(size):<12}"
            )

        else:

            missing.append(target)

            print(f"{red('MISSING'):<19}" f"{target:<38}" f"{'-':<12}" f"{'-':<12}")

    print()

    print(f"Regression trained: " f"{len(trained)}/{len(REGRESSION_TARGETS)}")

    print(f"Regression missing: " f"{len(missing)}")

    print(f"Regression model storage: " f"{human_size(total_size)}")

    if not missing:

        success("ALL REGRESSION MODELS AVAILABLE")

    else:

        warning("Some regression models are missing")

    return {
        "trained": trained,
        "missing": missing,
        "storage": total_size,
    }


# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================


def evaluate_classification_models():

    print()

    separator()

    print(bold("4. CLASSIFICATION MODELS"))

    separator()

    trained = []

    missing = []

    total_size = 0

    print()

    print(f"Expected classification models: " f"{len(CLASSIFICATION_TARGETS)}")

    print()

    print(f"{'MODEL':<10} " f"{'TARGET':<38} " f"{'OVERALL ACCURACY':<18} " f"{'CLASS CHECK':<18}")

    print("-" * 78)

    for target in CLASSIFICATION_TARGETS:

        path = model_path(target)

        metadata = load_json(metadata_path(target))

        if path.exists():

            size = file_size(path)

            total_size += size

            trained.append(target)

            metrics = metadata.get("metrics", {}) if metadata else {}
            accuracy = metrics.get("accuracy") if metrics else None

            classes = metadata.get("classes", []) if metadata else []
            report = metrics.get("classification_report", {}) if metrics else {}
            mapping = metadata.get("mapping", {}) if metadata else {}
            low_recall_classes = []

            for label, class_index in mapping.items():

                class_metrics = report.get(str(class_index), {})
                recall = class_metrics.get("recall")

                if isinstance(recall, (int, float)) and recall < 0.60:

                    low_recall_classes.append(label)

            accuracy_text = percentage(accuracy)
            class_text = (
                f"Review {len(low_recall_classes)} (<60%)"
                if low_recall_classes
                else f"All {len(classes)} >=60%" if classes else "N/A"
            )

            print(
                f"{green('READY'):<19}"
                f"{target:<38}"
                f"{accuracy_text:<18}"
                f"{class_text:<10}"
            )

        else:

            missing.append(target)

            print(f"{red('MISSING'):<19}" f"{target:<38}" f"{'-':<18}" f"{'-':<18}")

    print()

    print(f"Classification trained: " f"{len(trained)}/{len(CLASSIFICATION_TARGETS)}")

    print(f"Classification missing: " f"{len(missing)}")

    print(f"Classification model storage: " f"{human_size(total_size)}")

    if not missing:

        success("ALL CLASSIFICATION MODELS AVAILABLE")

    else:

        warning("Some classification models are missing")

    return {
        "trained": trained,
        "missing": missing,
        "storage": total_size,
    }


# =============================================================================
# CLASSIFICATION DETAILS
# =============================================================================


def evaluate_classification_details():

    print()

    separator()

    print(bold("5. CLASSIFICATION DETAILS"))

    separator()

    for target in CLASSIFICATION_TARGETS:

        metadata = load_json(metadata_path(target))

        if not metadata:

            continue

        classes = metadata.get(
            "classes",
            [],
        )

        mapping = metadata.get("mapping", {})

        metrics = metadata.get("metrics", {})
        accuracy = metrics.get("accuracy") if metrics else None

        print()

        print(bold(target.replace('_', ' ').title()))

        print(f"  Overall validation accuracy: {percentage(accuracy)}")

        print(
            f"  Training rows: " f"{metadata.get('training_rows', 'N/A'):,}"
            if isinstance(
                metadata.get("training_rows"),
                int,
            )
            else f"  Training rows: " f"{metadata.get('training_rows', 'N/A')}"
        )

        print(
            f"  Validation rows: " f"{metadata.get('validation_rows', 'N/A'):,}"
            if isinstance(
                metadata.get("validation_rows"),
                int,
            )
            else f"  Validation rows: " f"{metadata.get('validation_rows', 'N/A')}"
        )

        report = metrics.get("classification_report", {}) if metrics else {}

        if report:

            print("  Class results (recall is class-wise accuracy):")
            print(f"    {'CLASS':<16} {'CLASS ACC.':>11} {'PRECISION':>11} {'F1 SCORE':>10} {'SAMPLES':>10} {'REVIEW':>10}")
            index_to_label = {str(index): label for label, index in mapping.items()}

            if any(str(label).strip().upper() == "HIGH" for label in index_to_label.values()):

                print("    Class label HIGH is hidden from this breakdown; overall accuracy includes every class.")

            print("    Any displayed class accuracy below 60% is marked REVIEW.")

            for class_index, label in index_to_label.items():

                if str(label).strip().upper() == "HIGH":

                    continue

                class_metrics = report.get(class_index)

                if not class_metrics:

                    continue

                precision = class_metrics.get("precision")
                recall = class_metrics.get("recall")
                f1 = class_metrics.get("f1-score")
                support = class_metrics.get("support", 0)
                class_accuracy = recall
                review = "REVIEW" if isinstance(class_accuracy, (int, float)) and class_accuracy < 0.60 else "OK"

                print(
                    f"    {str(label):<16} "
                    f"{percentage(class_accuracy):>11} "
                    f"{percentage(precision):>11} "
                    f"{percentage(f1):>10} "
                    f"{support:>10,.0f} "
                    f"{review:>10}"
                )

            for average_name in ("macro avg", "weighted avg"):

                average_metrics = report.get(average_name)

                if average_metrics:

                    print(
                        f"    {average_name:<16} "
                        f"{percentage(average_metrics.get('recall')):>11} "
                        f"{percentage(average_metrics.get('precision')):>11} "
                        f"{percentage(average_metrics.get('f1-score')):>10} "
                        f"{average_metrics.get('support', 0):>10,.0f}"
                    )


# =============================================================================
# CHECKPOINT STATUS
# =============================================================================


def evaluate_checkpoints():

    print()

    separator()

    print(bold("6. CHECKPOINT STATUS"))

    separator()

    if not CHECKPOINT_ROOT.exists():

        warning("Checkpoint directory does not exist")

        return

    completed = []

    incomplete = []

    for target in ALL_TARGETS:

        path = checkpoint_path(target)

        if not path.exists():

            incomplete.append(target)

            continue

        data = load_json(path)

        if data and data.get("status") == "completed":

            completed.append(target)

        else:

            incomplete.append(target)

    print()

    print(f"Completed checkpoints: " f"{len(completed)}/{len(ALL_TARGETS)}")

    if completed:

        print()

        for target in completed:

            print(f"  [OK] {target}")

    if incomplete:

        print()

        print("Missing/incomplete checkpoints:")

        for target in incomplete:

            print(f"  - {target}")


# =============================================================================
# TRAINING SUMMARY
# =============================================================================


def evaluate_training_summary(dataset_status=None):

    print()

    separator()

    print(bold("7. TRAINING SUMMARY"))

    separator()

    if not TRAINING_SUMMARY_FILE.exists():

        warning("Training summary not found")

        return None

    summary = load_json(TRAINING_SUMMARY_FILE)

    if not summary:

        failure("Could not read training summary")

        return None

    print()

    print(f"Version: " f"{summary.get('version', 'N/A')}")

    status = "FAILED TARGETS PRESENT" if summary.get("failed") else "NO FAILED TARGETS"
    print(f"Status: {status}")

    dataset_rows = summary.get("dataset_rows")
    print(f"Dataset rows: {dataset_rows:,}" if isinstance(dataset_rows, int) else "Dataset rows: N/A")

    parquet_rows = dataset_status.get("rows") if dataset_status else None

    if isinstance(parquet_rows, int) and isinstance(dataset_rows, int):

        if parquet_rows == dataset_rows:

            print("Dataset rows skipped: 0")

        else:

            print(f"Parquet rows not included in that training run: {max(0, parquet_rows - dataset_rows):,}")

    print(f"Regression targets: " f"{summary.get('regression_models', 'N/A')}")

    print(f"Classification targets: " f"{summary.get('classification_models', 'N/A')}")

    print(f"Total targets: " f"{summary.get('total_models', 'N/A')}")

    runtime = summary.get("runtime_hours")

    if isinstance(
        runtime,
        (int, float),
    ):

        print(f"Training runtime: " f"{runtime:.2f} hours")

    completed = summary.get("completed_this_run", [])

    skipped = summary.get("skipped_existing", [])

    failed = summary.get("failed", [])

    print()

    print(f"Targets trained this run: {len(completed)}")

    print(f"Existing model files reused: {len(skipped)}")


    print(f"Failed targets: " f"{len(failed)}")

    if failed:

        print()

        warning("Failed targets:")

        for target in failed:

            print(f"  - {target}")

    return summary


# =============================================================================
# STORAGE STATUS
# =============================================================================


def evaluate_storage():

    print()

    separator()

    print(bold("8. SOALM STORAGE"))

    separator()

    dataset_size = file_size(TRAINING_FILE)

    model_size = directory_size(MODEL_ROOT)

    checkpoint_size = directory_size(CHECKPOINT_ROOT)

    log_size = directory_size(LOG_ROOT)

    total = dataset_size + model_size + checkpoint_size + log_size

    print()

    print(f"Training dataset: " f"{human_size(dataset_size)}")

    print(f"Models + metadata: " f"{human_size(model_size)}")

    print(f"Checkpoints: " f"{human_size(checkpoint_size)}")

    print(f"Logs: " f"{human_size(log_size)}")

    print()

    print(bold(f"Total V2 storage: " f"{human_size(total)}"))

    return {
        "dataset": dataset_size,
        "models": model_size,
        "checkpoints": checkpoint_size,
        "logs": log_size,
        "total": total,
    }


# =============================================================================
# OLD / OTHER SOALM MODELS
# =============================================================================


def evaluate_other_models():

    print()

    separator()

    print(bold("9. OTHER SOALM MODEL FILES"))

    separator()

    possible_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data" / "models",
    ]

    found = []

    for root in possible_roots:

        if not root.exists():

            continue

        try:

            for path in root.rglob("*.joblib"):

                # Don't duplicate V2 files.
                if MODEL_ROOT in path.parents:

                    continue

                found.append(path)

        except Exception:

            pass

    # Remove duplicates.
    found = list(dict.fromkeys(found))

    if not found:

        print()

        print("No other .joblib files found.")

        return

    print()

    print(f"Other .joblib files: " f"{len(found)}")

    print()

    for path in found:

        print(f"  {human_size(file_size(path)):<12} " f"{path}")


# =============================================================================
# FINAL READINESS
# =============================================================================


def final_status(
    dataset_status,
    regression_status,
    classification_status,
):

    print()

    separator()

    print(bold("10. FINAL SOALM V2 STATUS"))

    separator()

    dataset_ok = (
        dataset_status.get(
            "exists",
            False,
        )
        and not dataset_status.get(
            "missing_features",
            [],
        )
        and not dataset_status.get(
            "missing_targets",
            [],
        )
    )

    regression_complete = len(regression_status["missing"]) == 0

    classification_complete = len(classification_status["missing"]) == 0

    print()

    if dataset_ok:

        success("Training dataset is valid")

    else:

        failure("Training dataset has problems")

    if regression_complete:

        success(
            f"Regression: "
            f"{len(REGRESSION_TARGETS)}/"
            f"{len(REGRESSION_TARGETS)} trained"
        )

    else:

        warning(
            f"Regression: "
            f"{len(regression_status['trained'])}/"
            f"{len(REGRESSION_TARGETS)} trained"
        )

    if classification_complete:

        success(
            f"Classification: "
            f"{len(CLASSIFICATION_TARGETS)}/"
            f"{len(CLASSIFICATION_TARGETS)} trained"
        )

    else:

        warning(
            f"Classification: "
            f"{len(classification_status['trained'])}/"
            f"{len(CLASSIFICATION_TARGETS)} trained"
        )

    print()

    total_expected = len(REGRESSION_TARGETS) + len(CLASSIFICATION_TARGETS)

    total_trained = len(regression_status["trained"]) + len(
        classification_status["trained"]
    )

    print(bold(f"TOTAL MODELS: " f"{total_trained}/{total_expected}"))

    print()

    if dataset_ok and regression_complete and classification_complete:

        print(green("========================================"))

        print(green("       SOALM V2 - FULLY TRAINED"))

        print(green("========================================"))

        print()

        print("SOALM V2 has both:")

        print("  [OK] Rainfall regression intelligence")

        print("  [OK] Risk/event classification intelligence")

    elif total_trained > 0:

        print(yellow("SOALM V2 - PARTIALLY TRAINED"))

        print()

        print("Missing models must be trained before")

        print("calling the V2 intelligence layer complete.")

    else:

        print(red("SOALM V2 - NOT TRAINED"))


# =============================================================================
# MAIN
# =============================================================================


def main():

    configure_console()

    print_header()

    dataset_status = evaluate_dataset()

    evaluate_feature_metadata()

    regression_status = evaluate_regression_models()

    classification_status = evaluate_classification_models()

    evaluate_classification_details()

    evaluate_checkpoints()

    evaluate_training_summary(dataset_status)

    evaluate_storage()

    evaluate_other_models()

    final_status(
        dataset_status,
        regression_status,
        classification_status,
    )

    print()

    separator()

    print(bold("SOALM V2 EVALUATION COMPLETE"))

    separator()

    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
