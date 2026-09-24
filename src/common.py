"""Shared paths, column definitions, guards and results I/O."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_JSON = RESULTS_DIR / "results.json"
CURVES_JSON = RESULTS_DIR / "curves.json"

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT = "treatment"
OUTCOMES = ["visit", "conversion"]  # primary first
PRIMARY_OUTCOME = "visit"
POST_TREATMENT = ["exposure"]  # never a feature: it is observed after assignment

SEED = 42


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Fail loudly if a post-treatment or outcome column is used as a model feature."""
    cols = set(columns)
    forbidden = cols & set(POST_TREATMENT + OUTCOMES)
    assert not forbidden, f"Leakage: forbidden columns used as features: {sorted(forbidden)}"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def update_results(section: str, payload: dict, path: Path = RESULTS_JSON) -> None:
    """Merge `payload` into results.json under `section`, keeping other sections intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(section, {}).update(_to_jsonable(payload))
    path.write_text(json.dumps(data, indent=2))


def load_results(path: Path = RESULTS_JSON) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}
