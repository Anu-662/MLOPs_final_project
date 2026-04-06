"""Acceptance test cases for the prediction API."""

from model_acceptance_tests.test_case_schema import TestCase, TestCaseInput

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

case_paris_apartment = TestCase(
    name="Paris apartment 50m2 3 rooms",
    input=TestCaseInput(
        surface_reelle_bati=50.0,
        nombre_pieces_principales=3.0,
        code_departement="75",
        type_local="Appartement",
    ),
)

case_house_rhone = TestCase(
    name="House Rhone 100m2 5 rooms",
    input=TestCaseInput(
        surface_reelle_bati=100.0,
        nombre_pieces_principales=5.0,
        code_departement="69",
        type_local="Maison",
    ),
)

# ---------------------------------------------------------------------------
# Additional test cases
# ---------------------------------------------------------------------------

case_paris_studio = TestCase(
    name="Small Paris studio 20m2 1 room",
    input=TestCaseInput(
        surface_reelle_bati=20.0,
        nombre_pieces_principales=1.0,
        code_departement="75",
        type_local="Appartement",
    ),
)

case_large_house_var = TestCase(
    name="Large house Var 120m2 6 rooms",
    input=TestCaseInput(
        surface_reelle_bati=120.0,
        nombre_pieces_principales=6.0,
        code_departement="83",
        type_local="Maison",
    ),
)

case_dependency_bdrhone = TestCase(
    name="Dependency Bouches-du-Rhone 40m2",
    input=TestCaseInput(
        surface_reelle_bati=40.0,
        nombre_pieces_principales=1.0,
        code_departement="13",
        type_local="Dépendance",
    ),
)

ACCEPTANCE_TEST_CASES = [
    case_paris_apartment,
    case_house_rhone,
    case_paris_studio,
    case_large_house_var,
    case_dependency_bdrhone,
]
