"""Pfade und Modell-Erkennung.

Modelle liegen außerhalb des Projekts (mehrere GB). Standard ist die vorhandene Installation unter
~/tools/imagegen/models; überschreibbar mit der Umgebungsvariable T2I_MODELS_DIR.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("T2I_DATA_DIR", ROOT / "data"))
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "t2i.sqlite"
OV_CACHE_DIR = DATA_DIR / "ov_cache"
MODELS_DIR = Path(os.environ.get("T2I_MODELS_DIR", Path.home() / "tools" / "imagegen" / "models"))
DEVICE = os.environ.get("T2I_DEVICE", "GPU")

# Sinnvolle Standardwerte je Modellfamilie (aus den Messungen in Riddle_Games/docs/recherche-bildgenerierung.md)
FAMILY_DEFAULTS = {
    "flux": {"steps": 4, "guidance": 0.0, "negative": False},
    "sdxl": {"steps": 25, "guidance": 6.0, "negative": True},
    "sd": {"steps": 25, "guidance": 7.0, "negative": True},
    "sd3": {"steps": 28, "guidance": 5.0, "negative": True},
}


def _family(model_dir: Path) -> str | None:
    index = model_dir / "model_index.json"
    if not index.exists():
        return None
    cls = json.loads(index.read_text(encoding="utf-8")).get("_class_name", "")
    if "Flux" in cls:
        return "flux"
    if "XL" in cls:
        return "sdxl"
    if "StableDiffusion3" in cls:
        return "sd3"
    if "StableDiffusion" in cls or "LatentConsistency" in cls:
        return "sd"
    return None


def discover_models() -> dict[str, dict]:
    """Alle OpenVINO-Diffusionsmodelle im Modellordner: name -> {path, family, defaults}."""
    models = {}
    if not MODELS_DIR.is_dir():
        return models
    for d in sorted(MODELS_DIR.iterdir()):
        fam = _family(d) if d.is_dir() else None
        if fam:
            models[d.name] = {"path": str(d), "family": fam, "defaults": FAMILY_DEFAULTS[fam]}
    return models


for _d in (DATA_DIR, IMAGES_DIR, OV_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
