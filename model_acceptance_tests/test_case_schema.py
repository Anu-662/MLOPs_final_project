from pydantic import BaseModel, Field

class TestCaseInput(BaseModel):
    surface_reelle_bati: float
    nombre_pieces_principales: float
    code_departement: str
    code_postal: str = Field(default="00000", description="Postal code (optional, defaults to 00000)")
    type_local: str

class TestCase(BaseModel):
    """One test case: input and optional expected value or expected status."""

    name: str = Field(..., description="Short name for the case")
    input: TestCaseInput
    expected_value_eur: float | None = Field(None, description="If set, response must match (with tolerance)")
    expected_status: int | None = Field(None, description="If set, HTTP status must match (e.g. 422)")
