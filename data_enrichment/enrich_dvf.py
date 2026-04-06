"""
# v2
Data enrichment for DVF CSV files.

Enriches raw DVF data with validated and derived features:
  1. Validates and cleans code_postal (fills missing, fixes format)
  2. Extracts code_departement from code_postal when dept is missing
  3. Computes price_per_m2 for each transaction
  4. Computes department-level average price per m² (for price adequacy)
  5. Generates a department stats JSON file for the API to use

Run from repo root:
    python -m data_enrichment.enrich_dvf --input data/ --output data_enriched/

Or use as a library:
    from data_enrichment.enrich_dvf import enrich_dvf_dataframe
    enriched = enrich_dvf_dataframe(raw_df)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

def validate_code_postal(ser: pd.Series) -> pd.Series:
    """Validate and clean postal codes.

    Rules:
      - Must be exactly 5 digits (e.g. "75015", "69001")
      - "None", "nan", empty strings → NaN
      - Codes that don't match 5-digit format → NaN
      - Leading zeros are preserved (e.g. "01000" for Ain)

    Returns a cleaned Series with invalid codes set to NaN.
    """

    def clean_one(val) -> str | float:
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.lower() in ("none", "nan", ""):
            return np.nan
        # Remove any non-digit characters
        digits = "".join(c for c in s if c.isdigit())
        if len(digits) == 5:
            return digits
        # Some codes might be 4 digits (missing leading zero)
        if len(digits) == 4:
            return "0" + digits
        return np.nan

    return ser.map(clean_one)

def extract_dept_from_postal(postal: str) -> str | float:
    """Extract department code from postal code.

    French postal codes: first 2 digits = department (except Corsica: 20 → 2A/2B).
    For DOM-TOM: first 3 digits (e.g. 971 = Guadeloupe).

    Examples:
      "75015" → "75"
      "69001" → "69"
      "01000" → "01"
      "97100" → "971"
    """
    if pd.isna(postal):
        return np.nan
    s = str(postal).strip()
    if len(s) < 2:
        return np.nan
    # DOM-TOM: 3-digit department codes
    if s.startswith("97") and len(s) >= 3:
        return s[:3]
    return s[:2]

def fill_missing_dept_from_postal(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing code_departement using code_postal where possible.

    If a row has a valid postal code but no department code, we can derive
    the department from the first 2 digits of the postal code.
    """
    if "code_postal" not in df.columns or "code_departement" not in df.columns:
        return df

    df = df.copy()
    mask = df["code_departement"].isna() | (df["code_departement"].astype(str).str.strip() == "") | (df["code_departement"].astype(str).str.lower() == "none")
    valid_postal = df["code_postal"].notna()

    fill_mask = mask & valid_postal
    if fill_mask.any():
        df.loc[fill_mask, "code_departement"] = df.loc[fill_mask, "code_postal"].map(extract_dept_from_postal)
        print(f"  Filled {fill_mask.sum()} missing department codes from postal codes.")

    return df

def compute_price_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Add a price_per_m2 column (valeur_fonciere / surface_reelle_bati).

    Rows with zero or missing surface get NaN (we can't compute price/m²).
    This column is useful for analysis and for computing department averages.
    """
    df = df.copy()
    surface = df["surface_reelle_bati"]
    value = df["valeur_fonciere"]

    # Avoid division by zero
    valid = (surface.notna()) & (surface > 0) & (value.notna()) & (value > 0)
    df["price_per_m2"] = np.nan
    df.loc[valid, "price_per_m2"] = value[valid] / surface[valid]

    return df

def compute_department_stats(df: pd.DataFrame) -> dict:
    """Compute average price per m² for each department.

    Returns a dict like:
      {
        "75": {"avg_price_per_m2": 10500.0, "median_price_per_m2": 10200.0, "count": 42},
        "69": {"avg_price_per_m2": 3800.0, "median_price_per_m2": 3600.0, "count": 15},
        ...
      }

    This can replace the hardcoded DEPT_AVG_PRICE_PER_M2 in app.py for a more
    data-driven price adequacy label.
    """
    if "price_per_m2" not in df.columns or "code_departement" not in df.columns:
        return {}

    # Only use rows with valid price_per_m2
    valid = df[df["price_per_m2"].notna() & (df["price_per_m2"] > 0)]

    stats = {}
    for dept, group in valid.groupby("code_departement"):
        dept_str = str(dept).strip()
        if not dept_str or dept_str.lower() == "nan":
            continue
        stats[dept_str] = {
            "avg_price_per_m2": round(group["price_per_m2"].mean(), 2),
            "median_price_per_m2": round(group["price_per_m2"].median(), 2),
            "count": int(len(group)),
        }

    return stats

def enrich_dvf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run all enrichment steps on a DVF DataFrame.

    Steps (in order):
      1. Validate and clean code_postal
      2. Fill missing code_departement from code_postal
      3. Compute price_per_m2

    Returns the enriched DataFrame with a report printed to stdout.
    """
    report = {}
    report["input_rows"] = len(df)

    #  Validate code_postal
    if "code_postal" in df.columns:
        before_valid = df["code_postal"].notna().sum()
        df["code_postal"] = validate_code_postal(df["code_postal"])
        after_valid = df["code_postal"].notna().sum()
        report["postal_cleaned"] = int(before_valid - after_valid)
        report["postal_valid"] = int(after_valid)
    else:
        report["postal_cleaned"] = 0
        report["postal_valid"] = 0
        print("  Warning: no code_postal column found in data.")

    # Fill missing department codes
    df = fill_missing_dept_from_postal(df)

    #Compute price per m²
    df = compute_price_per_m2(df)
    report["price_per_m2_computed"] = int(df["price_per_m2"].notna().sum())

    # Print report
    print("\n── Data enrichment report ────────────────────────────────")
    print(f"  Input rows:              {report['input_rows']}")
    print(f"  Postal codes cleaned:    {report['postal_cleaned']} invalid → NaN")
    print(f"  Postal codes valid:      {report['postal_valid']}")
    print(f"  Price/m² computed:       {report['price_per_m2_computed']} rows")
    print("──────────────────────────────────────────────────────────\n")

    return df

def main() -> None:
    """CLI entry point: enrich all CSVs in data/ and write to data_enriched/."""
    import argparse

    parser = argparse.ArgumentParser(description="Enrich DVF data for CESAR training")
    parser.add_argument("--input", default="data", help="Input directory with raw CSVs")
    parser.add_argument("--output", default="data_enriched", help="Output directory for enriched CSV")
    parser.add_argument("--separator", default=";", help="CSV separator (default: ;)")
    parser.add_argument("--stats-output", default=None, help="Path to write department stats JSON (optional)")
    args = parser.parse_args()

    # Load all CSVs
    input_dir = Path(args.input)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    frames = [pd.read_csv(f, sep=args.separator, low_memory=False) for f in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} file(s), {len(combined)} raw rows.")

    # Clean the data first if Member B's cleaning module is available.
    # This removes duplicates, outliers, and missing values before enrichment
    # so that department stats are computed on clean data.
    try:
        from data_validation.clean_dvf import clean_dvf_dataframe
        print("Cleaning data before enrichment ...")
        combined = clean_dvf_dataframe(combined)
    except ImportError:
        print("Warning: data_validation.clean_dvf not found, enriching raw data.")

    # Enrich
    enriched = enrich_dvf_dataframe(combined)

    # Save enriched CSV
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dvf_enriched.csv"
    enriched.to_csv(out_path, sep=args.separator, index=False)
    print(f"Saved enriched data to {out_path}")

    # Compute and save department stats
    stats = compute_department_stats(enriched)
    if stats:
        stats_path = Path(args.stats_output) if args.stats_output else out_dir / "department_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved department stats to {stats_path}")
        print(f"  Departments found: {len(stats)}")
        for dept, s in sorted(stats.items()):
            print(f"    {dept}: avg {s['avg_price_per_m2']:,.0f} €/m² "
                  f"(median {s['median_price_per_m2']:,.0f}, n={s['count']})")
    else:
        print("  No department stats computed (missing data).")

if __name__ == "__main__":
    main()
