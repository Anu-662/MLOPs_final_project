from typing import Optional

from pydantic import BaseModel, Field

# Response schema is the API contract for clients: they know exactly which fields to expect and
# their types. Adding optional fields (e.g. value_low_eur) with default None keeps backward
# compatibility: old clients ignore them, new clients can use them.
class EstimateResponse(BaseModel):
    estimated_value_eur: float = Field(..., description="Estimated value in euros")

    # Confidence interval bounds (not yet implemented — placeholder for future quantile regression).
    value_low_eur: Optional[float] = Field(None, description="Lower bound (optional)")
    value_high_eur: Optional[float] = Field(None, description="Upper bound (optional)")

    # Price adequacy fields (added in v2).
    # These help the user understand whether the estimated price is reasonable
    # compared to the local market average for that department.
    price_per_m2: Optional[float] = Field(
        None, description="Estimated price per square metre"
    )
    price_adequacy: Optional[str] = Field(
        None,
        description="Price adequacy label: underpriced, fair, or overpriced",
    )
