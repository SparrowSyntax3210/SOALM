"""
===============================================================================
SOALM V2 FINAL — 40 FEATURE / 30 MODEL TRAINING
===============================================================================

PURPOSE
-------
Fresh/resumable final training of SOALM V2 using:

    40 FEATURES
    19 REGRESSION TARGETS
    11 CLASSIFICATION TARGETS
    30 TOTAL MODELS

DATASET
-------
data/era5/prepared/soalm_v2_5m.parquet

OUTPUT DIRECTORIES
------------------
data/era5/models_v2_final/
data/era5/checkpoints_v2_final/
data/era5/logs_v2_final/
data/era5/features_v2_final/

IMPORTANT
---------
This script DOES NOT use:

    models_v2_5m
    checkpoints_v2_5m
    logs_v2_5m

RESUME BEHAVIOUR
----------------
If a model already exists in:

    data/era5/models_v2_final/

then that model is SKIPPED.

Existing final models are NEVER deleted or overwritten.

Only missing models are trained.

ARCHITECTURE
------------
The trained models form the SOALM intelligence layer used by:

    1. Spatial Engine
       -> Predict risk across geographical grid cells

    2. What-If Simulation
       -> Modify weather features and predict future outcomes

    3. Disaster AI Chat
       -> Query model predictions and provide explanations

The models themselves are prediction models.
The Spatial Engine, What-If Engine and AI Chat operate above them.

===============================================================================
"""

from pathlib import Path
import os
import gc
import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "data" / "era5" / "prepared" / "soalm_v2_5m.parquet"

MODEL_DIR = BASE_DIR / "data" / "era5" / "models_v2_final"

CHECKPOINT_DIR = BASE_DIR / "data" / "era5" / "checkpoints_v2_final"

LOG_DIR = BASE_DIR / "data" / "era5" / "logs_v2_final"

FEATURE_DIR = BASE_DIR / "data" / "era5" / "features_v2_final"


# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

VERSION = "SOALM V2 FINAL"

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_DATASET_ROWS = 5_000_000


# =============================================================================
# RANDOM FOREST CONFIGURATION
# =============================================================================

REGRESSION_ESTIMATORS = 40

CLASSIFICATION_ESTIMATORS = 40

MAX_DEPTH = 14

MIN_SAMPLES_LEAF = 2

MAX_FEATURES = 0.35

N_JOBS = 4


# =============================================================================
# FEATURES — EXACTLY 40
# =============================================================================

FEATURES = [
    # -------------------------------------------------------------------------
    # Atmospheric state
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Historical rainfall
    # -------------------------------------------------------------------------
    "rain_last_1h",
    "rain_last_3h",
    "rain_last_6h",
    "rain_last_12h",
    "rain_last_24h",
    "rain_last_48h",
    "rain_last_72h",
    # -------------------------------------------------------------------------
    # Atmospheric changes
    # -------------------------------------------------------------------------
    "pressure_change_1h",
    "pressure_change_3h",
    "pressure_change_6h",
    "temperature_change_3h",
    "temperature_change_6h",
    "humidity_change_3h",
    "humidity_change_6h",
    "wind_change_3h",
    "wind_change_6h",
    # -------------------------------------------------------------------------
    # Spatial
    # -------------------------------------------------------------------------
    "latitude",
    "longitude",
    "elevation",
    # -------------------------------------------------------------------------
    # Temporal
    # -------------------------------------------------------------------------
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    # -------------------------------------------------------------------------
    # Cyclic time encoding
    # -------------------------------------------------------------------------
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]


# =============================================================================
# REGRESSION TARGETS — 19
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


# =============================================================================
# CLASSIFICATION TARGETS — 11
# =============================================================================

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


# =============================================================================
# ALL TARGETS
# =============================================================================

ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS


# =============================================================================
# VALIDATION
# =============================================================================


def validate_configuration():

    print()
    print("=" * 80)
    print("VALIDATING SOALM V2 FINAL CONFIGURATION")
    print("=" * 80)

    print()
    print("Features:", len(FEATURES))
    print(
        "Regression targets:",
        len(REGRESSION_TARGETS),
    )
    print(
        "Classification targets:",
        len(CLASSIFICATION_TARGETS),
    )
    print(
        "Total models:",
        len(ALL_TARGETS),
    )

    if len(FEATURES) != 40:

        raise RuntimeError(f"Expected 40 features, found {len(FEATURES)}")

    if len(REGRESSION_TARGETS) != 19:

        raise RuntimeError(
            f"Expected 19 regression targets, " f"found {len(REGRESSION_TARGETS)}"
        )

    if len(CLASSIFICATION_TARGETS) != 11:

        raise RuntimeError(
            f"Expected 11 classification targets, "
            f"found {len(CLASSIFICATION_TARGETS)}"
        )

    if len(ALL_TARGETS) != 30:

        raise RuntimeError(f"Expected 30 total targets, " f"found {len(ALL_TARGETS)}")

    print()
    print("✓ 40 features confirmed")
    print("✓ 19 regression targets confirmed")
    print("✓ 11 classification targets confirmed")
    print("✓ 30 models confirmed")


# =============================================================================
# HEADER
# =============================================================================


def print_header():

    print()

    print("=" * 80)
    print("SOALM V2 FINAL — RESUMABLE MODEL TRAINING")
    print("=" * 80)

    print()

    print("Dataset:")
    print(DATASET)

    print()

    print("MODEL DIRECTORY:")
    print(MODEL_DIR)

    print()

    print("CHECKPOINT DIRECTORY:")
    print(CHECKPOINT_DIR)

    print()

    print("LOG DIRECTORY:")
    print(LOG_DIR)

    print()

    print("FEATURE DIRECTORY:")
    print(FEATURE_DIR)

    print()

    print("Dataset limit:")
    print(f"{MAX_DATASET_ROWS:,} rows")

    print()

    print("Configuration:")

    print(
        "  Regression trees:",
        REGRESSION_ESTIMATORS,
    )

    print(
        "  Classification trees:",
        CLASSIFICATION_ESTIMATORS,
    )

    print(
        "  Max depth:",
        MAX_DEPTH,
    )

    print(
        "  Min samples leaf:",
        MIN_SAMPLES_LEAF,
    )

    print(
        "  Max features:",
        MAX_FEATURES,
    )

    print(
        "  CPU workers:",
        N_JOBS,
    )

    print(
        "  Validation split:",
        TEST_SIZE,
    )

    print()

    print("CPU available:")
    print(os.cpu_count())

    print()

    print("=" * 80)


# =============================================================================
# MODEL PATH
# =============================================================================


def model_file(target):

    return MODEL_DIR / f"{target}.joblib"


# =============================================================================
# CHECKPOINT PATH
# =============================================================================


def checkpoint_file(target):

    return CHECKPOINT_DIR / f"{target}.json"


# =============================================================================
# CHECK WHETHER MODEL ALREADY EXISTS
# =============================================================================


def model_already_exists(target):

    output = model_file(target)

    return output.exists() and output.stat().st_size > 0


# =============================================================================
# LOAD DATASET
# =============================================================================


def load_dataset():

    print()
    print("=" * 80)
    print("LOADING SOALM V2 5M DATASET")
    print("=" * 80)

    if not DATASET.exists():

        raise FileNotFoundError(f"\nDataset not found:\n{DATASET}\n")

    print()

    print("Disk size:")

    print(
        round(
            DATASET.stat().st_size / (1024**3),
            2,
        ),
        "GB",
    )

    print()

    print("Reading Parquet...")

    df = pd.read_parquet(
        DATASET,
        engine="pyarrow",
    )

    print()

    print("✓ Dataset loaded")

    print()

    print(
        "Rows:",
        f"{len(df):,}",
    )

    print(
        "Columns:",
        len(df.columns),
    )

    # -------------------------------------------------------------------------
    # Dataset cap
    # -------------------------------------------------------------------------

    if len(df) > MAX_DATASET_ROWS:

        print()

        print("WARNING: Dataset contains more than " f"{MAX_DATASET_ROWS:,} rows.")

        print(f"Using first {MAX_DATASET_ROWS:,} rows.")

        df = df.iloc[:MAX_DATASET_ROWS].copy()

    # -------------------------------------------------------------------------
    # Validate columns
    # -------------------------------------------------------------------------

    required = FEATURES + ALL_TARGETS

    missing = [column for column in required if column not in df.columns]

    if missing:

        raise RuntimeError("Missing required columns:\n" + "\n".join(missing))

    print()

    print("✓ 40 features verified")
    print("✓ 30 targets verified")

    # -------------------------------------------------------------------------
    # Numeric optimization
    # -------------------------------------------------------------------------

    print()

    print("Optimizing numeric memory...")

    for column in FEATURES:

        if pd.api.types.is_float_dtype(df[column]):

            df[column] = df[column].astype("float32")

        elif pd.api.types.is_integer_dtype(df[column]):

            df[column] = pd.to_numeric(
                df[column],
                downcast="integer",
            )

    print("✓ Numeric memory optimized")

    return df


# =============================================================================
# PREPARE FEATURES
# =============================================================================


def prepare_features(df):

    print()
    print("=" * 80)
    print("PREPARING 40 FEATURE MATRIX")
    print("=" * 80)

    X = df[FEATURES].copy()

    # -------------------------------------------------------------------------
    # Infinity
    # -------------------------------------------------------------------------

    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    print()
    print("Checking missing values...")

    missing_summary = {}

    for column in FEATURES:

        missing_count = int(X[column].isna().sum())

        missing_summary[column] = missing_count

        if missing_count > 0:

            median = X[column].median()

            if pd.isna(median):

                median = 0.0

            X[column] = X[column].fillna(median)

    # -------------------------------------------------------------------------
    # float32
    # -------------------------------------------------------------------------

    for column in FEATURES:

        if pd.api.types.is_numeric_dtype(X[column]):

            X[column] = X[column].astype("float32")

    print()

    print("Feature matrix:")
    print(X.shape)

    print()

    print(
        "Feature memory:",
        round(
            X.memory_usage(deep=True).sum() / (1024**3),
            2,
        ),
        "GB",
    )

    # -------------------------------------------------------------------------
    # Save feature metadata
    # -------------------------------------------------------------------------

    feature_metadata = {
        "version": VERSION,
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "dataset": str(DATASET),
        "missing_values_filled": missing_summary,
        "random_state": RANDOM_STATE,
    }

    metadata_file = FEATURE_DIR / "soalm_v2_features.json"

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_metadata,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Save pickle for API
    # -------------------------------------------------------------------------

    feature_pickle = FEATURE_DIR / "soalm_v2_features.pkl"

    joblib.dump(
        FEATURES,
        feature_pickle,
        compress=3,
    )

    print()

    print("✓ Feature metadata saved")
    print(metadata_file)

    print()

    print("✓ Feature list saved")
    print(feature_pickle)

    return X


# =============================================================================
# CLASSIFICATION LABEL ENCODING
# =============================================================================


def encode_labels(series):

    # -------------------------------------------------------------------------
    # Numeric
    # -------------------------------------------------------------------------

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    non_null = int(series.notna().sum())

    if numeric.notna().sum() == non_null:

        unique = sorted(numeric.dropna().unique().tolist())

        mapping = {value: index for index, value in enumerate(unique)}

        encoded = numeric.map(mapping)

        return (
            encoded,
            [str(x) for x in unique],
            {str(k): int(v) for k, v in mapping.items()},
        )

    # -------------------------------------------------------------------------
    # String
    # -------------------------------------------------------------------------

    values = series.astype("string")

    unique = sorted(values.dropna().unique().tolist())

    mapping = {value: index for index, value in enumerate(unique)}

    encoded = values.map(mapping)

    return (
        encoded,
        [str(x) for x in unique],
        {str(k): int(v) for k, v in mapping.items()},
    )


# =============================================================================
# TRAIN REGRESSION
# =============================================================================


def train_regression(
    target,
    X,
    df,
    model_index,
):

    print()
    print("=" * 80)

    print(f"REGRESSION MODEL " f"[{model_index}/{len(REGRESSION_TARGETS)}]")

    print()

    print("TARGET:")
    print(target)

    print()

    print("=" * 80)

    # -------------------------------------------------------------------------
    # SKIP EXISTING MODEL
    # -------------------------------------------------------------------------

    output = model_file(target)

    if model_already_exists(target):

        print()
        print("✓ MODEL ALREADY EXISTS")
        print("Skipping training.")
        print(output)

        return "skipped"

    start = time.time()

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    y = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    valid = y.notna()

    valid_count = int(valid.sum())

    print()

    print(
        "Valid rows:",
        f"{valid_count:,}",
    )

    if valid_count < 100:

        raise RuntimeError(f"Not enough valid rows for {target}")

    # -------------------------------------------------------------------------
    # Select valid data
    # -------------------------------------------------------------------------

    X_target = X.loc[valid]

    y_target = y.loc[valid].astype("float32")

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_target,
        y_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print()

    print(
        "Training rows:",
        f"{len(X_train):,}",
    )

    print(
        "Validation rows:",
        f"{len(X_val):,}",
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print()
    print("Training RandomForestRegressor...")

    model = RandomForestRegressor(
        n_estimators=REGRESSION_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=True,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    print()
    print("✓ Regression training complete")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    print()
    print("Evaluating...")

    prediction = model.predict(X_val)

    mae = mean_absolute_error(
        y_val,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            prediction,
        )
    )

    r2 = r2_score(
        y_val,
        prediction,
    )

    elapsed = time.time() - start

    print()
    print("-" * 60)

    print(
        "MAE :",
        round(mae, 6),
    )

    print(
        "RMSE:",
        round(rmse, 6),
    )

    print(
        "R²  :",
        round(r2, 6),
    )

    print(
        "Time:",
        round(elapsed / 60, 2),
        "minutes",
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": target,
        "type": "regression",
        "version": VERSION,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
    }

    print()
    print("Saving model...")

    joblib.dump(
        artifact,
        output,
        compress=3,
    )

    print()
    print("✓ MODEL SAVED")
    print(output)

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    checkpoint = {
        "target": target,
        "type": "regression",
        "status": "completed",
        "version": VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "training_seconds": float(elapsed),
    }

    with open(
        checkpoint_file(target),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    del model
    del X_target
    del y_target
    del X_train
    del X_val
    del y_train
    del y_val
    del prediction
    del y

    gc.collect()

    return "completed"


# =============================================================================
# TRAIN CLASSIFICATION
# =============================================================================


def train_classification(
    target,
    X,
    df,
    model_index,
):

    print()
    print("=" * 80)

    print(f"CLASSIFICATION MODEL " f"[{model_index}/{len(CLASSIFICATION_TARGETS)}]")

    print()

    print("TARGET:")
    print(target)

    print()

    print("=" * 80)

    # -------------------------------------------------------------------------
    # SKIP EXISTING MODEL
    # -------------------------------------------------------------------------

    output = model_file(target)

    if model_already_exists(target):

        print()
        print("✓ MODEL ALREADY EXISTS")
        print("Skipping training.")
        print(output)

        return "skipped"

    start = time.time()

    # -------------------------------------------------------------------------
    # Encode labels
    # -------------------------------------------------------------------------

    encoded, classes, mapping = encode_labels(df[target])

    valid = encoded.notna()

    X_target = X.loc[valid]

    y_target = encoded.loc[valid].astype("int8")

    print()

    print(
        "Valid rows:",
        f"{len(y_target):,}",
    )

    print()
    print("Classes:")

    for class_index, class_name in enumerate(classes):

        count = int((y_target == class_index).sum())

        percentage = count / len(y_target) * 100

        print(
            f"  {class_index}: " f"{class_name} " f"{count:,} " f"({percentage:.3f}%)"
        )

    if len(y_target) < 100:

        raise RuntimeError(f"Not enough rows for {target}")

    # -------------------------------------------------------------------------
    # One-class target
    # -------------------------------------------------------------------------

    if y_target.nunique() < 2:

        print()
        print("⚠ Only one class detected.")
        print("Using DummyClassifier.")

        X_train, X_val, y_train, y_val = train_test_split(
            X_target,
            y_target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        model = DummyClassifier(strategy="most_frequent")

        model.fit(
            X_train,
            y_train,
        )

    else:

        # ---------------------------------------------------------------------
        # Stratified split
        # ---------------------------------------------------------------------

        X_train, X_val, y_train, y_val = train_test_split(
            X_target,
            y_target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_target,
        )

        print()

        print(
            "Training rows:",
            f"{len(X_train):,}",
        )

        print(
            "Validation rows:",
            f"{len(X_val):,}",
        )

        # ---------------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------------

        print()
        print("Training RandomForestClassifier...")

        model = RandomForestClassifier(
            n_estimators=CLASSIFICATION_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            bootstrap=True,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )

        model.fit(
            X_train,
            y_train,
        )

    print()
    print("✓ Classification training complete")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    prediction = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        prediction,
    )

    elapsed = time.time() - start

    print()
    print("-" * 60)

    print(
        "Accuracy:",
        round(
            accuracy * 100,
            3,
        ),
        "%",
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    try:

        report = classification_report(
            y_val,
            prediction,
            output_dict=True,
            zero_division=0,
        )

    except Exception:

        report = {}

    # -------------------------------------------------------------------------
    # Save artifact
    # -------------------------------------------------------------------------

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": target,
        "type": "classification",
        "version": VERSION,
        "classes": classes,
        "mapping": mapping,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "metrics": {
            "accuracy": float(accuracy),
            "classification_report": report,
        },
    }

    print()
    print("Saving model...")

    joblib.dump(
        artifact,
        output,
        compress=3,
    )

    print()
    print("✓ MODEL SAVED")
    print(output)

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    checkpoint = {
        "target": target,
        "type": "classification",
        "status": "completed",
        "version": VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "mapping": mapping,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "metrics": {
            "accuracy": float(accuracy),
            "classification_report": report,
        },
        "training_seconds": float(elapsed),
    }

    with open(
        checkpoint_file(target),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    del model
    del X_target
    del y_target
    del X_train
    del X_val
    del y_train
    del y_val
    del prediction
    del encoded

    gc.collect()

    return "completed"


# =============================================================================
# SAVE MODEL REGISTRY
# =============================================================================


def save_model_registry():

    registry = {
        "version": VERSION,
        "feature_count": len(FEATURES),
        "regression_model_count": len(REGRESSION_TARGETS),
        "classification_model_count": len(CLASSIFICATION_TARGETS),
        "total_model_count": len(ALL_TARGETS),
        "features": FEATURES,
        "regression_targets": REGRESSION_TARGETS,
        "classification_targets": CLASSIFICATION_TARGETS,
        "models": {target: str(model_file(target)) for target in ALL_TARGETS},
    }

    registry_file = MODEL_DIR / "model_registry.json"

    with open(
        registry_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            registry,
            f,
            indent=2,
        )

    print()
    print("✓ Model registry saved:")
    print(registry_file)


# =============================================================================
# TRAIN ALL
# =============================================================================


def train_all(df):

    print()
    print("=" * 80)
    print("STARTING RESUMABLE TRAINING — 30 MODELS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Check existing models BEFORE loading full feature matrix
    # -------------------------------------------------------------------------

    existing_models = [target for target in ALL_TARGETS if model_already_exists(target)]

    missing_models = [
        target for target in ALL_TARGETS if not model_already_exists(target)
    ]

    print()

    print(
        "Existing final models:",
        len(existing_models),
        "/",
        len(ALL_TARGETS),
    )

    print(
        "Models requiring training:",
        len(missing_models),
        "/",
        len(ALL_TARGETS),
    )

    if existing_models:

        print()
        print("MODELS THAT WILL BE SKIPPED:")

        for target in existing_models:

            print(
                "  ✓",
                target,
            )

    if not missing_models:

        print()
        print("✓ ALL 30 MODEL FILES ALREADY EXIST; reusing them without retraining.")

        save_model_registry()

        return {
            "completed": [],
            "skipped": existing_models,
            "failed": [],
        }

    # -------------------------------------------------------------------------
    # Prepare X
    # -------------------------------------------------------------------------

    X = prepare_features(df)

    completed = []
    skipped = list(existing_models)
    failed = []

    start_all = time.time()

    # =========================================================================
    # REGRESSION
    # =========================================================================

    for index, target in enumerate(
        REGRESSION_TARGETS,
        start=1,
    ):

        try:

            status = train_regression(
                target,
                X,
                df,
                index,
            )

            if status == "completed":

                completed.append(target)

            elif status == "skipped":

                if target not in skipped:

                    skipped.append(target)

        except Exception as e:

            print()
            print(
                "✗ REGRESSION FAILED:",
                target,
            )

            print(
                "ERROR:",
                repr(e),
            )

            traceback.print_exc()

            failed.append(target)

        gc.collect()

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    for index, target in enumerate(
        CLASSIFICATION_TARGETS,
        start=1,
    ):

        try:

            status = train_classification(
                target,
                X,
                df,
                index,
            )

            if status == "completed":

                completed.append(target)

            elif status == "skipped":

                if target not in skipped:

                    skipped.append(target)

        except Exception as e:

            print()
            print(
                "✗ CLASSIFICATION FAILED:",
                target,
            )

            print(
                "ERROR:",
                repr(e),
            )

            traceback.print_exc()

            failed.append(target)

        gc.collect()

    # =========================================================================
    # RUNTIME
    # =========================================================================

    total_seconds = time.time() - start_all

    # =========================================================================
    # MODEL REGISTRY
    # =========================================================================

    save_model_registry()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {
        "version": VERSION,
        "dataset": str(DATASET),
        "dataset_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "regression_models": len(REGRESSION_TARGETS),
        "classification_models": len(CLASSIFICATION_TARGETS),
        "total_models": len(ALL_TARGETS),
        "completed_this_run": completed,
        "skipped_existing": skipped,
        "failed": failed,
        "completed_count": len(completed),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "available_model_count": len(
            [target for target in ALL_TARGETS if model_already_exists(target)]
        ),
        "runtime_hours": (total_seconds / 3600),
        "configuration": {
            "regression_estimators": (REGRESSION_ESTIMATORS),
            "classification_estimators": (CLASSIFICATION_ESTIMATORS),
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": (MIN_SAMPLES_LEAF),
            "max_features": MAX_FEATURES,
            "n_jobs": N_JOBS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
    }

    summary_file = LOG_DIR / "training_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("SOALM V2 FINAL — TRAINING SUMMARY")
    print("=" * 80)

    print()

    print(
        "Trained this run:",
        len(completed),
        "/",
        len(ALL_TARGETS),
    )

    print(
        "Existing model files reused (not retrained):",
        len(skipped),
        "/",
        len(ALL_TARGETS),
    )

    print(
        "Failed:",
        len(failed),
        "/",
        len(ALL_TARGETS),
    )

    print()

    print(
        "Available final models:",
        len([target for target in ALL_TARGETS if model_already_exists(target)]),
        "/",
        len(ALL_TARGETS),
    )

    print()

    print(
        "Runtime:",
        round(
            total_seconds / 3600,
            2,
        ),
        "hours",
    )

    if failed:

        print()
        print("FAILED MODELS:")

        for target in failed:

            print(
                "  ✗",
                target,
            )

    print()
    print("Training summary:")
    print(summary_file)

    print()

    available_count = len(
        [target for target in ALL_TARGETS if model_already_exists(target)]
    )

    if available_count == len(ALL_TARGETS):

        print("✓ ALL 30 MODELS ARE AVAILABLE")

        print()
        print("SOALM V2 FINAL TRAINING " "SUCCESSFULLY COMPLETED.")

    else:

        print("⚠ TRAINING FINISHED " "WITHOUT ALL MODELS AVAILABLE.")

    print()

    print("=" * 80)

    del X

    gc.collect()

    return {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():

    print_header()

    process_start = time.time()

    try:

        # ---------------------------------------------------------------------
        # Validate configuration
        # ---------------------------------------------------------------------

        validate_configuration()

        # ---------------------------------------------------------------------
        # Resume information
        # ---------------------------------------------------------------------

        print()
        print("=" * 80)
        print("RESUMABLE TRAINING MODE")
        print("=" * 80)

        print()

        print("Existing models in " "models_v2_final will be SKIPPED.")

        print()

        print("Existing final models will " "NOT be deleted or overwritten.")

        print()

        print("Only missing models will " "be trained.")

        print()

        existing_count = len(
            [target for target in ALL_TARGETS if model_already_exists(target)]
        )

        missing_count = len(ALL_TARGETS) - existing_count

        print(
            "Already available:",
            existing_count,
            "/",
            len(ALL_TARGETS),
        )

        print(
            "Need training:",
            missing_count,
            "/",
            len(ALL_TARGETS),
        )

        print()

        print("Old directories remain untouched.")

        print("=" * 80)

        # ---------------------------------------------------------------------
        # Load dataset
        # ---------------------------------------------------------------------

        df = load_dataset()

        # ---------------------------------------------------------------------
        # Train
        # ---------------------------------------------------------------------

        result = train_all(df)

        # ---------------------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------------------

        del df

        gc.collect()

        # ---------------------------------------------------------------------
        # Final status
        # ---------------------------------------------------------------------

        elapsed = time.time() - process_start

        print()
        print("=" * 80)
        print("SOALM V2 FINAL — PROCESS FINISHED")
        print("=" * 80)

        print()

        print(
            "Total process time:",
            round(
                elapsed / 3600,
                2,
            ),
            "hours",
        )

        print()

        print("FINAL MODEL DIRECTORY:")

        print(MODEL_DIR)

        print()

        print("FEATURE DIRECTORY:")

        print(FEATURE_DIR)

        print()

        print("LOG DIRECTORY:")

        print(LOG_DIR)

        print()

        available_count = len(
            [target for target in ALL_TARGETS if model_already_exists(target)]
        )

        if available_count == len(ALL_TARGETS):

            print("✓ 30/30 MODELS AVAILABLE")

            print()

            print("SOALM V2 FINAL TRAINING " "SUCCESSFULLY COMPLETED.")

        else:

            print(f"⚠ {len(ALL_TARGETS) - available_count} " f"MODEL(S) STILL MISSING.")

        print()

        print("=" * 80)

    except KeyboardInterrupt:

        print()
        print("=" * 80)
        print("TRAINING INTERRUPTED")
        print("=" * 80)

        print()

        print("Models already saved in:")

        print(MODEL_DIR)

        print()

        print(
            "Rerunning this script will "
            "SKIP models that already exist "
            "and continue with missing models."
        )

    except Exception as e:

        print()
        print("=" * 80)
        print("FATAL ERROR")
        print("=" * 80)

        print()

        print(repr(e))

        traceback.print_exc()

        raise

    finally:

        gc.collect()

        elapsed = time.time() - process_start

        print()

        print(
            "Process runtime:",
            round(
                elapsed / 3600,
                2,
            ),
            "hours",
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":

    main()
