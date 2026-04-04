# Training pipeline: train on dataframe, export artifact + contract.
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# joblib is the recommended way to persist sklearn models (handles numpy arrays and large objects
# better than pickle). We write a versioned filename (e.g. model_20250101120000.joblib) so
# deployments can pin to a known version and roll back if needed; the contract file uses the same
# version so model and contract always match.
from prediction_contract.feature_schema import (
    TARGET_NAME,
    TYPE_LOCAL_CATEGORIES,
    MODEL_FEATURE_NAMES,
)
from prediction_contract.contract_version import ContractVersion


def _code_departement_to_numeric(ser: pd.Series) -> pd.Series:
    """Convert department code strings to floats. Handles Corsica (2A→20, 2B→21)."""

    def map_one(val: str) -> float:
        if pd.isna(val):
            return 0.0
        s = str(val).strip()
        if s == "2A":
            return 20.0
        if s == "2B":
            return 21.0
        try:
            return float(int(s))
        except ValueError:
            return 0.0

    return ser.map(map_one)


def _build_postal_code_target_map(df: pd.DataFrame) -> dict[str, float]:
    """Compute the average property price for each postal code.

    Target encoding replaces a categorical value (like "75006") with the average
    of the target variable (valeur_fonciere) for that category. This gives the
    model meaningful numeric information: 75006 -> 1,200,000 (expensive area)
    vs 75019 -> 400,000 (cheaper area).

    Why this works better than raw numeric encoding:
      - Raw: 75006 and 75019 are just big numbers with no price meaning
      - Target encoded: 75006 = 1.2M and 75019 = 400K — the model sees
        actual price differences between areas

    Returns a dict mapping postal code string to average price.
    """
    postal_str = df["code_postal"].astype(str).str.strip()
    target = df[TARGET_NAME]

    # Group by postal code, compute mean price
    mapping = {}
    for code, group in zip(postal_str, target):
        mapping.setdefault(code, []).append(group)

    return {code: np.mean(values) for code, values in mapping.items()}


def _apply_postal_target_encoding(ser: pd.Series, target_map: dict[str, float], global_mean: float) -> pd.Series:
    """Replace postal code strings with their target-encoded values.

    Postal codes seen during training get their average price.
    Unseen postal codes (or missing values) get the global mean price.
    This ensures the model can handle new postal codes at inference time.
    """
    def encode_one(val) -> float:
        if pd.isna(val):
            return global_mean
        s = str(val).strip()
        return target_map.get(s, global_mean)

    return ser.map(encode_one)


def build_feature_matrix(df: pd.DataFrame, postal_target_map: dict[str, float] | None = None) -> np.ndarray:
    """Build the design matrix from a DataFrame.

    Column order MUST match MODEL_FEATURE_NAMES in feature_schema.py:
      [surface, pieces, dept, postal, type_Appartement, type_Maison, ...]

    v2: code_postal is target-encoded — each postal code is replaced with the
    average property price in that area. This gives the model meaningful
    location information instead of arbitrary numbers.
    """
    surface = df["surface_reelle_bati"].fillna(0.0).astype(np.float64)
    pieces = df["nombre_pieces_principales"].fillna(0.0).astype(np.float64)
    dept = _code_departement_to_numeric(df["code_departement"].astype(str))

    # v2: target-encoded postal code
    # During training, postal_target_map is computed from the training data.
    # During inference, the saved map is loaded from the artifact.
    global_mean = df[TARGET_NAME].mean() if TARGET_NAME in df.columns else 0.0
    if postal_target_map is None:
        # Training mode: build the map from this data
        postal_target_map = _build_postal_code_target_map(df)

    postal = _apply_postal_target_encoding(df["code_postal"].astype(str), postal_target_map, global_mean)

    type_local = df["type_local"].fillna("Appartement").astype(str)
    encoder = OneHotEncoder(categories=[TYPE_LOCAL_CATEGORIES], sparse_output=False)
    type_encoded = encoder.fit_transform(type_local.values.reshape(-1, 1))

    return np.column_stack([surface.values, pieces.values, dept.values, postal.values, type_encoded])


def train_on_dataframe(df: pd.DataFrame) -> tuple[Any, dict[str, float]]:
    """Train the model and return both the model and the postal code target map.

    The target map is saved alongside the model so that inference can encode
    new postal codes consistently with how the model was trained.
    """
    postal_target_map = _build_postal_code_target_map(df)
    X = build_feature_matrix(df, postal_target_map)
    y = df[TARGET_NAME].values.astype(np.float64)
    reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    reg.fit(X, y)
    return reg, postal_target_map


def export_artifact(
    model: Any,
    artifact_dir: Path,
    model_version: str | None = None,
    postal_target_map: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    version = model_version or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_path = artifact_dir / f"model_{version}.joblib"
    contract_path = artifact_dir / f"contract_{version}.json"

    # Save model and postal target map together so inference can use the same encoding
    joblib.dump({"model": model, "postal_target_map": postal_target_map or {}}, model_path)

    contract = ContractVersion(
        model_version=version,
        feature_names=MODEL_FEATURE_NAMES,
        target_name=TARGET_NAME,
        type_local_categories=TYPE_LOCAL_CATEGORIES,
    )
    contract_path.write_text(json.dumps(contract.to_serializable(), indent=2), encoding="utf-8")

    return model_path, contract_path


# Columns every training CSV must have (order used when building the combined table).
# v2: added "code_postal" to the required columns list.
REQUIRED_TRAINING_COLUMNS_ORDERED = (
    list(MODEL_FEATURE_NAMES[:4]) + ["type_local", TARGET_NAME]
)
REQUIRED_TRAINING_COLUMNS = set(REQUIRED_TRAINING_COLUMNS_ORDERED)


def load_dvf_subset_csv(csv_path: Path, separator: str = ";") -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=separator, low_memory=False)
    missing = REQUIRED_TRAINING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def train_from_csv_and_export(
    csv_path: Path,
    artifact_dir: Path,
    model_version: str | None = None,
    separator: str = ";",
) -> tuple[Path, Path]:
    df = load_dvf_subset_csv(csv_path, separator=separator)
    model, postal_target_map = train_on_dataframe(df)
    return export_artifact(model, artifact_dir, model_version=model_version, postal_target_map=postal_target_map)


def load_all_csvs_from_dir(
    data_dir: Path,
    separator: str = ";",
) -> pd.DataFrame:
    """Load every CSV in data_dir, validate columns, and concatenate."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files in {data_dir}. Add at least one CSV with columns: {REQUIRED_TRAINING_COLUMNS}"
        )

    reference_columns: set[str] | None = None
    frames: list[pd.DataFrame] = []

    for path in csv_files:
        df = pd.read_csv(path, sep=separator, low_memory=False)
        columns = set(df.columns)

        missing = REQUIRED_TRAINING_COLUMNS - columns
        if missing:
            raise ValueError(
                f"File {path.name} is missing required columns: {missing}. "
                f"Required: {REQUIRED_TRAINING_COLUMNS}"
            )

        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            only_in_this = columns - reference_columns
            only_in_others = reference_columns - columns
            raise ValueError(
                f"File {path.name} has different columns than other files. "
                f"Only in this file: {only_in_this}. Only in others: {only_in_others}. "
                "All CSVs in data/ must have the same columns."
            )

        frames.append(df[REQUIRED_TRAINING_COLUMNS_ORDERED])

    return pd.concat(frames, ignore_index=True)
