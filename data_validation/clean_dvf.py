# DVF data cleaning module.
# Handles deduplication, missing values, and outlier removal for DVF datasets.

from pathlib import Path

import numpy as np
import pandas as pd

def load_raw_csvs(data_dir: Path, separator: str = ";") -> pd.DataFrame:
    """Load all CSVs from a directory into one DataFrame."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = [pd.read_csv(f, sep=separator, low_memory=False) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} file(s), {len(combined)} raw rows.")
    return combined

def clean_dvf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DVF data for model training. Removes duplicates, missing values, and outliers."""
    report = {}
    report["raw_rows"] = len(df)

    # Deduplicate multi-lot transactions (keep largest surface per mutation)
    if "id_mutation" in df.columns:
        before = len(df)
        df = df.sort_values("surface_reelle_bati", ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["id_mutation"], keep="first")
        report["duplicates_removed"] = before - len(df)
    else:
        report["duplicates_removed"] = 0

    # Drop rows with missing/zero surface
    before = len(df)
    df = df[df["surface_reelle_bati"].notna() & (df["surface_reelle_bati"] > 0)]
    report["missing_surface_dropped"] = before - len(df)

    # Drop rows with missing/zero valeur_fonciere
    before = len(df)
    df = df[df["valeur_fonciere"].notna() & (df["valeur_fonciere"] > 0)]
    report["missing_value_dropped"] = before - len(df)

    # Remove price outliers using IQR
    before = len(df)
    q1 = df["valeur_fonciere"].quantile(0.25)
    q3 = df["valeur_fonciere"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df["valeur_fonciere"] >= lower_bound) & (df["valeur_fonciere"] <= upper_bound)]
    report["outliers_removed"] = before - len(df)
    report["outlier_bounds"] = (round(lower_bound, 0), round(upper_bound, 0))

    # Fill missing nombre_pieces_principales
    filled = df["nombre_pieces_principales"].isna().sum()
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(1.0)
    report["pieces_filled"] = filled

    report["clean_rows"] = len(df)


    print("\n── Data cleaning report ──────────────────────────────────")
    print(f"  Raw rows:                {report['raw_rows']}")
    print(f"  Duplicates removed:      {report['duplicates_removed']}")
    print(f"  Missing surface dropped: {report['missing_surface_dropped']}")
    print(f"  Missing value dropped:   {report['missing_value_dropped']}")
    print(f"  Outliers removed:        {report['outliers_removed']} "
          f"(bounds: {report['outlier_bounds'][0]:,.0f}€ – {report['outlier_bounds'][1]:,.0f}€)")
    print(f"  Pieces filled (NaN→1):   {report['pieces_filled']}")
    print(f"  Clean rows:              {report['clean_rows']}")
    print("──────────────────────────────────────────────────────────\n")

    return df

def main() -> None:
    """CLI entry point for cleaning DVF data."""
    import argparse

    parser = argparse.ArgumentParser(description="Clean DVF data for CESAR training")
    parser.add_argument("--input", default="data", help="Input directory with raw CSVs")
    parser.add_argument("--output", default="data_cleaned", help="Output directory for cleaned CSV")
    parser.add_argument("--separator", default=";", help="CSV separator (default: ;)")
    args = parser.parse_args()

    raw = load_raw_csvs(Path(args.input), separator=args.separator)
    cleaned = clean_dvf_dataframe(raw)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dvf_cleaned.csv"
    cleaned.to_csv(out_path, sep=args.separator, index=False)
    print(f"Saved cleaned data to {out_path}")

if __name__ == "__main__":
    main()