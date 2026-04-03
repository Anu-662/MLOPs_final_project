# Endpoint for department price stats (avg price/m²)
# Shows hardcoded averages + enriched stats if available

import json
from pathlib import Path

from fastapi import FastAPI

def register_stats_endpoint(app: FastAPI, dept_averages: dict[str, float]) -> None:
    """Register /stats/departments endpoint."""

    @app.get("/stats/departments")
    def get_department_stats() -> dict:
        """Return dept avg prices and adequacy thresholds."""
        response = {
            "source": "hardcoded",
            "note": "These are approximate values. Data-driven values from enrichment are shown separately if available.",
            "departments": {},
        }

        # Build dept info
        for dept, avg in sorted(dept_averages.items()):
            response["departments"][dept] = {
                "avg_price_per_m2": avg,
                "adequacy_thresholds": {
                    "underpriced_below": round(avg * 0.85, 2),
                    "overpriced_above": round(avg * 1.15, 2),
                },
            }

        # Load enriched stats if present
        enriched_stats_path = Path("data_enriched/department_stats.json")
        if enriched_stats_path.exists():
            try:
                with open(enriched_stats_path, "r", encoding="utf-8") as f:
                    data_driven = json.load(f)
                response["data_driven_stats"] = data_driven
                response["note"] = (
                    "Hardcoded values are used for predictions. "
                    "Data-driven values from the enrichment step are shown for comparison."
                )
            except (json.JSONDecodeError, OSError):
                pass

        return response
