from typing import Any

import numpy as np

from prediction_contract.request_schema import EstimateRequest
from prediction_contract.response_schema import EstimateResponse
from prediction_contract.contract_version import ContractVersion

# Raising a dedicated exception for invalid features lets the API return 422 and the CLI print a
# clear message instead of a generic traceback. The contract defines the allowed type_local
# categories; we must use the same order for one-hot encoding as in training.
class InvalidFeatureError(Exception):
    pass


def _code_departement_to_numeric(code: str) -> float:
    s = str(code).strip()
    if s == "2A":
        return 20.0
    if s == "2B":
        return 21.0
    try:
        return float(int(s))
    except ValueError:
        return 0.0


def request_to_feature_row(request: EstimateRequest, contract: ContractVersion,
                           postal_target_map: dict[str, float] | None = None,
                           global_mean: float = 0.0) -> np.ndarray:
    """Convert an API request into a feature row matching the training column order.

    Column order (must match MODEL_FEATURE_NAMES):
      [surface, pieces, dept, postal_encoded, type_Appartement, type_Maison, ...]

    v2: code_postal is target-encoded — replaced with the average property price
    for that postal code area. If the postal code wasn't seen during training,
    we use the global mean price as a fallback.
    """
    categories = contract.type_local_categories
    if request.type_local not in categories:
        raise InvalidFeatureError(f"type_local must be one of {categories}, got {request.type_local!r}")

    dept_num = _code_departement_to_numeric(request.code_departement)

    # Target-encode the postal code: look up the average price for this area
    # If we have a target map (from training), use it. Otherwise fall back to 0.0
    postal_str = str(request.code_postal).strip()
    if postal_target_map:
        postal_encoded = postal_target_map.get(postal_str, global_mean)
    else:
        # Fallback: raw numeric encoding (for backward compatibility)
        try:
            postal_encoded = float(int(postal_str))
        except ValueError:
            postal_encoded = 0.0

    type_one_hot = [1.0 if c == request.type_local else 0.0 for c in categories]

    ordered = [
        float(request.surface_reelle_bati),
        float(request.nombre_pieces_principales),
        dept_num,
        postal_encoded,
        *type_one_hot,
    ]
    return np.array(ordered, dtype=np.float64).reshape(1, -1)


def estimate_from_model(model_artifact: Any, request: EstimateRequest, contract: ContractVersion) -> EstimateResponse:
    """Run prediction using the model artifact.

    The artifact can be either:
      - A plain sklearn model (v1 / backward compatible)
      - A dict with {"model": sklearn_model, "postal_target_map": dict} (v2 with target encoding)
    """
    # Unpack the artifact — handle both old (plain model) and new (dict) formats
    if isinstance(model_artifact, dict):
        model = model_artifact["model"]
        postal_target_map = model_artifact.get("postal_target_map", {})
        # Compute global mean from the target map values (average of all area averages)
        global_mean = np.mean(list(postal_target_map.values())) if postal_target_map else 0.0
    else:
        model = model_artifact
        postal_target_map = None
        global_mean = 0.0

    X = request_to_feature_row(request, contract, postal_target_map, global_mean)
    pred = model.predict(X)
    value = float(pred.flat[0])
    return EstimateResponse(estimated_value_eur=value)
