# Prediction history endpoint for monitoring API usage
import csv
import os
from pathlib import Path

from fastapi import FastAPI, Query

DEFAULT_LOG_PATH = Path("logs/api_requests.csv")

def _get_log_path() -> Path:
    return Path(os.environ.get("CESAR_LOG_PATH", str(DEFAULT_LOG_PATH)))

def register_history_endpoint(app: FastAPI) -> None:
    """Register the /predictions/history endpoint on the given app."""

    @app.get("/predictions/history")
    def get_prediction_history(
        limit: int = Query(default=50, ge=1, le=500, description="Number of entries to return"),
        endpoint_filter: str = Query(default="/estimate/", description="Filter by endpoint"),
    ) -> dict:
        """Return recent prediction logs from the CSV log file."""
        log_path = _get_log_path()

        if not log_path.exists():
            return {"total_logged": 0, "returned": 0, "entries": []}

        entries = []
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if endpoint_filter and row.get("path", "") != endpoint_filter:
                    continue
                entries.append(row)

        # Most recent first
        entries.reverse()

        return {
            "total_logged": len(entries),
            "returned": min(limit, len(entries)),
            "entries": entries[:limit],
        }
