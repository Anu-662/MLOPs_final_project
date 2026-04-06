from pathlib import Path

from training.asset_rating_model.train_and_export import (
    load_all_csvs_from_dir,
    train_on_dataframe,
    export_artifact,
)

# Where to find data and where to write the model (relative to repo root).
DATA_DIR_NAME = "data"
CLEANED_DIR_NAME = "data_cleaned"
ARTIFACT_DIR_NAME = "artifact_storage"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / DATA_DIR_NAME
    cleaned_dir = repo_root / CLEANED_DIR_NAME
    artifact_dir = repo_root / ARTIFACT_DIR_NAME

    # Step 1: Clean the raw data if the cleaning module is available.
    # This ensures we always train on deduplicated, outlier-free data.
    try:
        from data_validation.clean_dvf import load_raw_csvs, clean_dvf_dataframe
        print(f"Cleaning raw data from {data_dir} ...")
        raw = load_raw_csvs(data_dir)
        cleaned = clean_dvf_dataframe(raw)

        # Save cleaned data so other scripts can use it too
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = cleaned_dir / "dvf_cleaned.csv"
        cleaned.to_csv(cleaned_path, sep=";", index=False)
        print(f"  Saved cleaned data to {cleaned_path}")

        # Train on cleaned data
        print("Training model on cleaned data ...")
        combined = cleaned
    except ImportError:
        # If cleaning module not available, fall back to raw data
        print("Warning: data_validation.clean_dvf not found, training on raw data.")
        print(f"Loading CSVs from {data_dir} ...")
        combined = load_all_csvs_from_dir(data_dir, separator=";")

    num_rows = len(combined)
    print(f"  Training on {num_rows} rows.")

    # Step 2: Train the model.
    # v2: train_on_dataframe returns both the model and the postal code target map
    print("Training model ...")
    model, postal_target_map = train_on_dataframe(combined)

    # Step 3: Save model, postal target map, and contract.
    model_path, contract_path = export_artifact(
        model,
        artifact_dir,
        model_version="minimal",
        postal_target_map=postal_target_map,
    )
    print(f"  Model:  {model_path}")
    print(f"  Contract: {contract_path}")

    print(
        "Set CESAR_MODEL_PATH and CESAR_CONTRACT_PATH to these paths "
        "(or symlink as model_latest.joblib / contract_latest.json)."
    )


if __name__ == "__main__":
    main()
