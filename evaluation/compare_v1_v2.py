"""
Model evaluation: compare v1 (without code_postal) vs v2 (with target-encoded code_postal).

Uses 5-fold cross-validation to measure MAE and RMSE for both versions,
then prints a comparison table.

Run from repo root:
    python -m evaluation.compare_v1_v2

What it does:
  1. Loads and cleans the DVF data (using clean_dvf).
  2. Builds feature matrix v1: [surface, pieces, dept, type_one_hot]
  3. Builds feature matrix v2: [surface, pieces, dept, target_encoded_postal, type_one_hot]
  4. Trains RandomForest on each using 5-fold cross-validation.
  5. Reports MAE, RMSE, and whether v2 is an improvement.

Note on target encoding:
  v2 replaces each postal code with the average property price in that area.
  To avoid data leakage during cross-validation, we compute the target encoding
  on the training folds only and apply it to the test fold. This is done using
  a custom cross-validation loop instead of sklearn's cross_val_score.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from prediction_contract.feature_schema import (
    TARGET_NAME,
    TYPE_LOCAL_CATEGORIES,
)


def _code_to_numeric(ser: pd.Series) -> np.ndarray:
    """Convert code strings to floats. Handles Corsica (2A→20, 2B→21)."""
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
    """v1 features: surface, pieces, dept, type_one_hot (NO code_postal)."""
    surface = df["surface_reelle_bati"].fillna(0).values.astype(np.float64)
    pieces = df["nombre_pieces_principales"].fillna(0).values.astype(np.float64)
    dept = _code_to_numeric(df["code_departement"])

    encoder = OneHotEncoder(categories=[TYPE_LOCAL_CATEGORIES], sparse_output=False)
    type_enc = encoder.fit_transform(df["type_local"].fillna("Appartement").values.reshape(-1, 1))

    return np.column_stack([surface, pieces, dept, type_enc])


def _target_encode_postal(train_postal: pd.Series, train_y: np.ndarray,
                          apply_postal: pd.Series, global_mean: float) -> np.ndarray:
    """Target-encode postal codes using training data only.

    For each postal code, compute the average target value from the training set.
    Apply that mapping to the given postal series. Unseen codes get the global mean.
    """
    # Build mapping from training data
    mapping = {}
    for code, price in zip(train_postal.astype(str).str.strip(), train_y):
        mapping.setdefault(code, []).append(price)
    target_map = {code: np.mean(prices) for code, prices in mapping.items()}

    # Apply to the given series
    encoded = apply_postal.astype(str).str.strip().map(
        lambda x: target_map.get(x, global_mean)
    ).values.astype(np.float64)

    return encoded


def evaluate_v1(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> dict:
    """Evaluate v1 with standard cross-validation (no target encoding needed)."""
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)

    mae_scores = []
    rmse_scores = []
    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        mae_scores.append(np.mean(np.abs(preds - y[test_idx])))
        rmse_scores.append(np.sqrt(np.mean((preds - y[test_idx]) ** 2)))

    return {
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
    }


def evaluate_v2_target_encoded(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """Evaluate v2 with target-encoded postal codes.

    Target encoding must be computed on training folds only to avoid data leakage.
    If we computed it on the full dataset, the model would "cheat" by seeing test
    data prices during encoding.
    """
    y = df[TARGET_NAME].values.astype(np.float64)
    surface = df["surface_reelle_bati"].fillna(0).values.astype(np.float64)
    pieces = df["nombre_pieces_principales"].fillna(0).values.astype(np.float64)
    dept = _code_to_numeric(df["code_departement"])
    postal_series = df["code_postal"].reset_index(drop=True)

    encoder = OneHotEncoder(categories=[TYPE_LOCAL_CATEGORIES], sparse_output=False)
    type_enc = encoder.fit_transform(df["type_local"].fillna("Appartement").values.reshape(-1, 1))

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)

    mae_scores = []
    rmse_scores = []

    for train_idx, test_idx in kf.split(surface):
        # Target-encode postal codes using ONLY training fold data
        global_mean = y[train_idx].mean()
        train_postal_encoded = _target_encode_postal(
            postal_series.iloc[train_idx], y[train_idx],
            postal_series.iloc[train_idx], global_mean
        )
        test_postal_encoded = _target_encode_postal(
            postal_series.iloc[train_idx], y[train_idx],
            postal_series.iloc[test_idx], global_mean
        )

        # Build feature matrices
        X_train = np.column_stack([
            surface[train_idx], pieces[train_idx], dept[train_idx],
            train_postal_encoded, type_enc[train_idx]
        ])
        X_test = np.column_stack([
            surface[test_idx], pieces[test_idx], dept[test_idx],
            test_postal_encoded, type_enc[test_idx]
        ])

        model.fit(X_train, y[train_idx])
        preds = model.predict(X_test)
        mae_scores.append(np.mean(np.abs(preds - y[test_idx])))
        rmse_scores.append(np.sqrt(np.mean((preds - y[test_idx]) ** 2)))

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
    results_v1 = evaluate_v1(X_v1, y)

    # v2: with target-encoded code_postal
    results_v2 = evaluate_v2_target_encoded(df)

    # Print comparison
    print("── Model evaluation report ───────────────────────────────")
    print(f"  Training rows:  {len(df)}")
    print(f"  Cross-val folds: {min(5, len(df))}")
    print(f"  Postal code encoding: target encoding (avg price per postal code)")
    print()
    print(f"  {'Metric':<12} {'v1 (no postal)':<24} {'v2 (target-enc postal)':<24} {'Better?'}")
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

    improvement_pct = (1 - results_v2["mae_mean"] / results_v1["mae_mean"]) * 100
    print()
    if improvement_pct > 0:
        print(f"  v2 reduces MAE by ~{improvement_pct:.1f}%.")
    else:
        print(f"  v2 increases MAE by ~{abs(improvement_pct):.1f}%. Target-encoded postal code did not help.")

    if len(df) < 100:
        print()
        print("  ⚠ Caveat: with few rows, these numbers have high variance.")
        print("  A larger dataset is needed for a reliable comparison.")


if __name__ == "__main__":
    main()
