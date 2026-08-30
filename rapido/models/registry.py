"""Persistence for trained models and their metrics."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

import config

logger = logging.getLogger(__name__)

METRICS_FILE = config.MODEL_DIR / "metrics.json"


def save_model(pipeline, name: str, metrics: dict, metadata: dict | None = None) -> Path:
    """Persist a fitted pipeline together with its metrics and metadata."""
    path = config.get_model_path(name)
    payload = {
        "pipeline": pipeline,
        "metrics": metrics,
        "metadata": {
            **(metadata or {}),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": name,
        },
    }
    joblib.dump(payload, path, compress=3)
    logger.info("Saved model %s to %s", name, path)
    save_metrics(name, {"metrics": metrics, **(metadata or {})})
    return path


def load_model(name: str):
    """Load a persisted pipeline and its metadata.

    Returns:
        Tuple of (pipeline, metadata dict including metrics).

    Raises:
        FileNotFoundError: If the artefact does not exist.
    """
    path = config.get_model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model named {name!r} at {path}. "
            "Run: python scripts/train_all.py"
        )
    payload = joblib.load(path)
    metadata = dict(payload.get("metadata", {}))
    metadata["metrics"] = payload.get("metrics", {})
    return payload["pipeline"], metadata


def model_exists(name: str) -> bool:
    """Return whether a trained artefact is on disk."""
    return config.get_model_path(name).exists()


def list_models() -> pd.DataFrame:
    """List every persisted model with its headline metric."""
    rows = []
    for path in sorted(config.MODEL_DIR.glob("*.joblib")):
        name = path.stem
        row = {
            "model": name,
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1_048_576, 2),
        }
        try:
            _, metadata = load_model(name)
            metrics = metadata.get("metrics", {})
            row["algorithm"] = metadata.get("algorithm", "-")
            row["headline"] = (
                metrics.get("f1_macro")
                or metrics.get("r2")
                or metrics.get("pr_auc")
                or "-"
            )
            row["trained_at"] = metadata.get("saved_at", "-")
        except Exception as exc:  # pragma: no cover - corrupt artefact
            row["algorithm"] = f"unreadable ({exc})"
        rows.append(row)
    return pd.DataFrame(rows)


def save_metrics(name: str, payload: dict) -> Path:
    """Merge one model's metrics into the shared metrics file."""
    existing = load_metrics() if METRICS_FILE.exists() else {}
    existing[name] = _jsonable(payload)
    METRICS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return METRICS_FILE


def load_metrics(name: str | None = None) -> dict:
    """Read stored metrics for one model, or all of them."""
    if not METRICS_FILE.exists():
        return {}
    data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return data.get(name, {}) if name else data


def _jsonable(value):
    """Recursively coerce numpy and pandas values into JSON-safe types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
