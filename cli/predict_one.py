"""CLI command: predict one property (single record).

The real work (load model, run estimate) lives in runtime.inference.
This file only defines the command-line interface (typer).

v2 change: added --postal flag for code_postal feature.
"""

import json
import os
from pathlib import Path

import typer

from prediction_contract.request_schema import EstimateRequest
from runtime.inference.estimate_from_artifact import estimate_from_model, InvalidFeatureError
from runtime.inference.load_artifact import load_artifact_from_path, ArtifactNotFoundError

predict_one_app = typer.Typer(help="Predict one property (single record).")

def _get_model_and_contract_paths(
    model_path: Path | None,
    contract_path: Path | None,
) -> tuple[Path, Path]:
    model = model_path or Path(os.environ.get("CESAR_MODEL_PATH", ""))
    contract = contract_path or Path(os.environ.get("CESAR_CONTRACT_PATH", ""))
    if not model or not model.exists():
        raise typer.BadParameter("Model path not set or file not found. Use --model or CESAR_MODEL_PATH.")
    if not contract or not contract.exists():
        raise typer.BadParameter("Contract path not set or file not found. Use --contract or CESAR_CONTRACT_PATH.")
    return model, contract

@predict_one_app.command("run")
def run_predict_one(
    surface_reelle_bati: float = typer.Option(..., "--surface", "-s", help="Living area in m²"),
    nombre_pieces_principales: float = typer.Option(..., "--pieces", "-p", help="Number of main rooms"),
    code_departement: str = typer.Option(..., "--departement", "-d", help="Department code (e.g. 75, 2A)"),
    code_postal: str = typer.Option("00000", "--postal", help="Postal code (e.g. 75015). Defaults to 00000."),
    type_local: str = typer.Option("Appartement", "--type", "-t", help="Property type"),
    model_path: Path | None = typer.Option(None, "--model", "-m", path_type=Path),
    contract_path: Path | None = typer.Option(None, "--contract", "-c", path_type=Path),
    json_input: Path | None = typer.Option(None, "--json", "-j", path_type=Path, help="Or read one record from JSON file"),
    output_json: bool = typer.Option(False, "--json-out", help="Output as JSON"),
) -> None:
    """Run the model on one property and print the estimated value."""
    if json_input is not None:
        raw = json.loads(json_input.read_text(encoding="utf-8"))
        request = EstimateRequest.model_validate(raw)
    else:
        request = EstimateRequest(
            surface_reelle_bati=surface_reelle_bati,
            nombre_pieces_principales=nombre_pieces_principales,
            code_departement=code_departement.strip()[:3],
            code_postal=code_postal.strip()[:5],
            type_local=type_local,
        )

    model_resolved, contract_resolved = _get_model_and_contract_paths(model_path, contract_path)
    try:
        model_obj, contract_obj = load_artifact_from_path(model_resolved, contract_resolved)
    except ArtifactNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    try:
        response = estimate_from_model(model_obj, request, contract_obj)
    except InvalidFeatureError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if output_json:
        typer.echo(response.model_dump_json(indent=2))
    else:
        typer.echo(f"estimated_value_eur: {response.estimated_value_eur}")
