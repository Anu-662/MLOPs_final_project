# Compare model v1 (without code_postal) vs v2 (with code_postal) using cross-validation.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

# Reuse project constants for consistency
from prediction_contract.feature_schema import (
    TARGET_NAME,
    TYPE_LOCAL_CATEGORIES,
)

def _code_to_numeric(ser: pd.Series) -> np.ndarray:
    """Convert department/postal code strings to numeric values."""

    def convert(val):
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

    return ser.map(convert).values.astype(np.float64)

def build_feature_matrix_v1(df: pd.DataFrame) -> np.ndarray:
    """Build v1 features: surface, pieces, dept, type (no postal code)."""
    surface = df["surface_reelle_bati"].fillna(0).values.astype(np.float64)
    pieces = df["nombre_pieces_principales"].fillna(0).values.astype(np.float64)
    dept = _code_to_numeric(df["code_departement"])

    encoder = OneHotEncoder(categories=[TYPE_LOCAL_CATEGORIES], sparse_output=False)
    type_enc = encoder.fit_transform(df["type_local"].fillna("Appartement").values.reshape(-1, 1))

    return np.column_stack([surface, pieces, dept, type_enc])

def build_feature_matrix_v2(df: pd.DataFrame) -> np.ndarray:
    """Build v2 features: surface, pieces, dept, postal, type."""
    surface = df["surface_reelle_bati"].fillna(0).values.astype(np.float64)
    pieces = df["nombre_pieces_principales"].fillna(0).values.astype(np.float64)
    dept = _code_to_numeric(df["code_departement"])
    postal = _code_to_numeric(df["code_postal"])

    encoder = OneHotEncoder(categories=[TYPE_LOCAL_CATEGORIES], sparse_output=False)
    type_enc = encoder.fit_transform(df["type_local"].fillna("Appartement").values.reshape(-1, 1))

    return np.column_stack([surface, pieces, dept, postal, type_enc])

def evaluate_model(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """Run cross-validation and return MAE + RMSE metrics."""
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

    # cross_val_score returns negative MAE (sklearn convention)
    mae_scores = -cross_val_score(model, X, y, cv=min(n_folds, len(y)), scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=min(n_folds, len(y)), scoring="neg_mean_squared_error"))

    return {
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
    }

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    try:
        from data_validation.clean_dvf import load_raw_csvs, clean_dvf_dataframe
        raw = load_raw_csvs(data_dir)
        df = clean_dvf_dataframe(raw)
    except ImportError:
        print("Warning: data_validation.clean_dvf not found, using basic loading.")
        csv_files = sorted(data_dir.glob("*.csv"))
        frames = [pd.read_csv(f, sep=";", low_memory=False) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["surface_reelle_bati"].notna() & (df["surface_reelle_bati"] > 0)]
        df = df.drop_duplicates(subset=["id_mutation"], keep="first")

    if len(df) < 5:
        print(f"\nOnly {len(df)} clean rows — too few for meaningful cross-validation.")
        print("Add more CSV files to data/ and re-run for better results.")
        return

    y = df[TARGET_NAME].values.astype(np.float64)

    print(f"Evaluating on {len(df)} clean rows...\n")

    # v1: without code_postal
    X_v1 = build_feature_matrix_v1(df)
    results_v1 = evaluate_model(X_v1, y)

    # v2: with code_postal
    X_v2 = build_feature_matrix_v2(df)
    results_v2 = evaluate_model(X_v2, y)

    # Print comparison
    print("── Model evaluation report ───────────────────────────────")
    print(f"  Training rows:  {len(df)}")
    print(f"  Cross-val folds: {min(5, len(df))}")
    print()
    print(f"  {'Metric':<12} {'v1 (no postal)':<24} {'v2 (with postal)':<24} {'Better?'}")
    print(f"  {'─'*12} {'─'*24} {'─'*24} {'─'*10}")

    mae_better = "← v2" if results_v2["mae_mean"] < results_v1["mae_mean"] else "← v1"
    rmse_better = "← v2" if results_v2["rmse_mean"] < results_v1["rmse_mean"] else "← v1"

    print(f"  {'MAE':<12} "
          f"{results_v1['mae_mean']:>12,.0f} ± {results_v1['mae_std']:>6,.0f}   "
          f"{results_v2['mae_mean']:>12,.0f} ± {results_v2['mae_std']:>6,.0f}   "
          f"{mae_better}")
    print(f"  {'RMSE':<12} "
          f"{results_v1['rmse_mean']:>12,.0f} ± {results_v1['rmse_std']:>6,.0f}   "
          f"{results_v2['rmse_mean']:>12,.0f} ± {results_v2['rmse_std']:>6,.0f}   "
          f"{rmse_better}")
    print("──────────────────────────────────────────────────────────")

    # Summary
    improvement_pct = (1 - results_v2["mae_mean"] / results_v1["mae_mean"]) * 100
    print()
    if improvement_pct > 0:
        print(f"  v2 reduces MAE by ~{improvement_pct:.1f}%.")
    else:
        print(f"  v2 increases MAE by ~{abs(improvement_pct):.1f}%. code_postal did not help.")

    print()
    print("  Note: small dataset means high variance in these results.")
    print("  More data would give a more reliable comparison.")

if __name__ == "__main__":
    main()
