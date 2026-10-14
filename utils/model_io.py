"""Save and load model artifacts with an integrity check.

joblib/pickle files can execute arbitrary code when loaded, so an app should only
load artifacts that YOU produced. We record a SHA-256 fingerprint at training time
and refuse to load a file whose fingerprint no longer matches.
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn


class ModelLoadError(RuntimeError):
    """Raised when a model artifact is missing, modified, or incompatible."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model, model_path: Path, meta_path: Path, metadata: dict) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        **metadata,
        "artifact": model_path.name,
        "sha256": file_sha256(model_path),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def load_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise ModelLoadError(f"Metadata file not found: {meta_path}")
    return json.loads(meta_path.read_text())


def load_trusted_model(model_path: Path, meta_path: Path):
    """Return (model, metadata, warnings). Raises ModelLoadError on failure."""
    if not model_path.exists():
        raise ModelLoadError(
            f"Model file not found: {model_path}. Run train_model.py first."
        )
    meta = load_metadata(meta_path)
    if file_sha256(model_path) != meta.get("sha256"):
        raise ModelLoadError(
            "Model fingerprint does not match metadata. The file may have been "
            "replaced or corrupted, so it will not be loaded."
        )
    warnings = []
    if meta.get("sklearn_version") != sklearn.__version__:
        warnings.append(
            f"Model trained with scikit-learn {meta.get('sklearn_version')}, "
            f"running {sklearn.__version__}. Pin the training version in "
            "requirements.txt to avoid subtle behaviour changes."
        )
    try:
        model = joblib.load(model_path)
    except Exception as exc:  # corrupted file, missing class, version clash
        raise ModelLoadError(f"Could not load model: {exc}") from exc
    return model, meta, warnings
