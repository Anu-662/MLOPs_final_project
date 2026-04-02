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

def _code_postal_to_numeric(ser: pd.Series) -> pd.Series:
    """Convert postal code strings to floats.

    Postal codes like "75015" become 75015.0. Missing or invalid values become 0.0.
    This is a simple numeric encoding — the model can learn that nearby postal codes
    (e.g. 75015 vs 75016) may have similar price patterns.

    Limitation: this treats postal codes as continuous numbers, which means the model
    might think 75015 is "between" 75014 and 75016 in some meaningful way. A more
    sophisticated approach would be target encoding or embedding, but for a first
    enrichment this is simple and defensible.
    """

    def map_one(val: str) -> float:
        if pd.isna(val):
            return 0.0
        s = str(val).strip()
        try:
            return float(int(s))
        except ValueError:
            return 0.0

    return ser.map(map_one)

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the design matrix from a DataFrame.

    Column order MUST match MODEL_FEATURE_NAMES in feature_schema.py:
      [surface, pieces, dept, postal, type_Appartement, type_Maison, ...]

    v2 change: added code_postal between code_departement and the one-hot type columns.
    """
    surface = df["surface_reelle_bati"].fillna(0.0).astype(np.float64)
    pieces = df["nombre_pieces_principales"].fillna(0.0).astype(np.float64)
    dept = _code_departement_to_numeric(df["code_departement"].astype(str))

    # NEW in v2: postal code as a numeric feature
    postal = _code_postal_to_numeric(df["code_postal"].astype(str))

    type_local = df["type_local"].fillna("Appartement").astype(str)
    encoder = OneHotEncoder(categories=[TYPE_LOCAL_CATEGORIES], sparse_output=False)
    type_encoded = encoder.fit_transform(type_local.values.reshape(-1, 1))

    return np.column_stack([surface.values, pieces.values, dept.values, postal.values, type_encoded])

def train_on_dataframe(df: pd.DataFrame) -> Any:
    X = build_feature_matrix(df)
    y = df[TARGET_NAME].values.astype(np.float64)
    reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    reg.fit(X, y)
    return reg

def export_artifact(
    model: Any,
    artifact_dir: Path,
    model_version: str | None = None,
) -> tuple[Path, Path]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    version = model_version or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_path = artifact_dir / f"model_{version}.joblib"
    contract_path = artifact_dir / f"contract_{version}.json"

    joblib.dump(model, model_path)

    contract = ContractVersion(
        model_version=version,
        feature_names=MODEL_FEATURE_NAMES,
        target_name=TARGET_NAME,
        type_local_categories=TYPE_LOCAL_CATEGORIES,
    )
    contract_path.write_text(json.dumps(contract.to_serializable(), indent=2), encoding="utf-8")

    return model_path, contract_path


# Columns every training CSV must have (order used when building the combined table).
# v2 change: added "code_postal" to the required columns list.
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
    model = train_on_dataframe(df)
    return export_artifact(model, artifact_dir, model_version=model_version)

def load_all_csvs_from_dir(
    data_dir: Path,
    separator: str = ";",
) -> pd.DataFrame:
    """  Load every CSV in data_dir, check they all have the same columns and the required
    training columns, then concatenate. Raises if no CSVs, missing columns, or schema mismatch.."""
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
 # Keep only required columns in a fixed order.
        frames.append(df[REQUIRED_TRAINING_COLUMNS_ORDERED])

    return pd.concat(frames, ignore_index=True)
