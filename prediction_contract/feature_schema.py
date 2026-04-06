# Single source of truth for feature and target names used in training and in the contract.
# The design matrix column order must stay identical in train_and_export and in
# estimate_from_artifact; changing order here would break deployed models.

TARGET_NAME: str = "valeur_fonciere"

# Categories for type_local (order must be stable for one-hot encoding).
TYPE_LOCAL_CATEGORIES: list[str] = [
    "Appartement",
    "Maison",
    "Dépendance",
    "Local industriel. commercial ou assimilé",
]

# Order of columns in the design matrix (after encoding).
# v2 change: added "code_postal" between "code_departement" and the type_local one-hot columns.
# Why: code_postal is more granular than code_departement (e.g. 75015 vs 75016 have very
# different property prices). The DVF data already contains this column.
MODEL_FEATURE_NAMES: list[str] = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "code_departement",
    "code_postal",  # added  code_postal feature in v2
    *[f"type_local_{c}" for c in TYPE_LOCAL_CATEGORIES],
]
