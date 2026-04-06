# OpenAPI docs config for the prediction API
from fastapi import FastAPI

# Example payloads for Swagger "Try it out"
ESTIMATE_EXAMPLE_REQUEST = {
    "summary": "Paris apartment — 50m², 3 rooms, 75015",
    "value": {
        "surface_reelle_bati": 50.0,
        "nombre_pieces_principales": 3.0,
        "code_departement": "75",
        "code_postal": "75015",
        "type_local": "Appartement",
    },
}

ESTIMATE_EXAMPLE_RESPONSE = {
    "estimated_value_eur": 523000.0,
    "value_low_eur": None,
    "value_high_eur": None,
    "price_per_m2": 10460.0,
    "price_adequacy": "fair",
}

def create_app() -> FastAPI:
    """Create FastAPI app with docs metadata."""
    app = FastAPI(
        title="CESAR Prediction API",
        version="0.2.0",
        description=(
            "**CESAR** (CentraleSupelec-ESSEC System for Asset Rating) estimates "
            "the market value of French properties using DVF open data.\n\n"
            "### Endpoints\n"
            "- **POST /estimate/** — Get a property valuation with price adequacy label\n"
            "- **GET /health** — Check if the model is loaded and ready\n"
            "- **GET /model_info** — See which model version is running\n\n"
            "### Price adequacy\n"
            "Each estimate includes a label indicating whether the price is "
            "`underpriced`, `fair`, or `overpriced` relative to the department average."
        ),
        openapi_tags=[
            {
                "name": "prediction",
                "description": "Property valuation endpoints",
            },
            {
                "name": "operations",
                "description": "Health checks and model metadata",
            },
        ],
    )
    return app
