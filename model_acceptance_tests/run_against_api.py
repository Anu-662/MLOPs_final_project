# Run acceptance tests against the API

import httpx

from model_acceptance_tests.test_case_schema import TestCase

def check_api_health(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Check if the API is healthy before running tests."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url.rstrip('/')}/health")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                return True, "API is healthy"
            return False, f"Unexpected health response: {data}"
        return False, f"Health check returned status {response.status_code}"
    except httpx.RequestError as e:
        return False, f"Could not reach API: {e}"

def run_one_case(
    base_url: str,
    case: TestCase,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Run a single test case and return pass/fail status."""
    payload = case.input.model_dump()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url.rstrip('/')}/estimate/", json=payload)

    # If expecting specific HTTP status, check that
    if case.expected_status is not None:
        if response.status_code != case.expected_status:
            return False, f"Expected status {case.expected_status}, got {response.status_code}"
        return True, "ok"

    if response.status_code != 200:
        return False, f"Status {response.status_code}: {response.text}"

    data = response.json()

    # Check response has estimated_value_eur
    value = data.get("estimated_value_eur")
    if value is None:
        return False, "Response missing estimated_value_eur"

    # Check value tolerance if expected
    if case.expected_value_eur is not None:
        tol = 0.01 * abs(case.expected_value_eur) or 1.0
        if abs(value - case.expected_value_eur) > tol:
            return False, f"Expected ~{case.expected_value_eur}, got {value}"

    # Validate price_adequacy label
    adequacy = data.get("price_adequacy")
    if adequacy is not None and adequacy not in ("underpriced", "fair", "overpriced"):
        return False, f"Unknown price_adequacy label: {adequacy!r}"

    return True, f"ok (value={value:,.0f}, adequacy={adequacy or 'n/a'})"

def run_all_cases(
    base_url: str,
    cases: list[TestCase],
    timeout: float = 10.0,
) -> list[tuple[str, bool, str]]:
    """Run all cases; return list of (name, passed, message)."""
    results: list[tuple[str, bool, str]] = []
    for case in cases:
        passed, msg = run_one_case(base_url, case, timeout=timeout)
        results.append((case.name, passed, msg))
    return results

def print_summary(results: list[tuple[str, bool, str]]) -> None:
    """Print test results summary table."""
    name_width = max(len(name) for name, _, _ in results) if results else 20

    print()
    print("── Acceptance test results ────────────────────────────────")
    print(f"  {'#':<3} {'Status':<7} {'Name':<{name_width}}  Message")

    for i, (name, passed, msg) in enumerate(results, 1):
        status = "PASS" if passed else "FAIL"
        print(f"  {i:<3} {status:<7} {name:<{name_width}}  {msg}")

    passed_count = sum(1 for _, p, _ in results if p)
    failed_count = len(results) - passed_count
    print("──────────────────────────────────────────────────────────")
    print(f"  Total: {len(results)} | Passed: {passed_count} | Failed: {failed_count}")
    print()

    if failed_count > 0:
        print("  Some tests failed. See messages above for details.")
