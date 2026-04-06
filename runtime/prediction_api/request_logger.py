"""
#  v2
Request logging middleware for the  CESAR API.

Logs every incoming request to a CSV file for monitoring and debugging.
Attach to the FastAPI app with: app.middleware("http")(request_logging_middleware)

Log file location: set CESAR_LOG_PATH env var (default: logs/api_requests.csv).
"""

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response

# Where to write the log file. Default: logs/api_requests.csv relative to working dir.
DEFAULT_LOG_PATH = Path("logs/api_requests.csv")

LOG_COLUMNS = [
    "timestamp",
    "method",
    "path",
    "status_code",
    "response_time_ms",
    "request_body",
    "client_ip",
]

def _get_log_path() -> Path:
    """Get the log file path from env var or use default."""
    return Path(os.environ.get("CESAR_LOG_PATH", str(DEFAULT_LOG_PATH)))

def _ensure_log_file(path: Path) -> None:
    """Create the log file with headers if it doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()

def _write_log_row(path: Path, row: dict) -> None:
    """Append one row to the log CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)

async def request_logging_middleware(request: Request, call_next) -> Response:
    """Middleware that logs every request to a CSV file.

    How to use:
        app.middleware("http")(request_logging_middleware)

    This runs for EVERY request (GET /health, POST /estimate/, etc.).
    It measures response time by recording timestamps before and after
    the request is processed.
    """
    log_path = _get_log_path()
    _ensure_log_file(log_path)

    # Record start time
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Read request body for POST requests (useful for debugging predictions)
    request_body = ""
    if request.method == "POST":
        try:
            body_bytes = await request.body()
            # Truncate to 500 chars to avoid huge log entries
            request_body = body_bytes.decode("utf-8", errors="replace")[:500]
        except Exception:
            request_body = "<could not read body>"

    # Process the request
    response = await call_next(request)

    # Calculate response time
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # Get client IP (handles proxied requests)
    client_ip = request.client.host if request.client else "unknown"

    # Write log entry
    try:
        _write_log_row(log_path, {
            "timestamp": timestamp,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "response_time_ms": elapsed_ms,
            "request_body": request_body,
            "client_ip": client_ip,
        })
    except Exception:
        # Never let logging errors crash the API
        pass

    return response
