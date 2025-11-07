import importlib
from pathlib import Path
import sys


def resolve_logic_class(dotted: str):
    mod_path, cls_name = dotted.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)

def project_root(markers: tuple[str, ...] = ("pyproject.toml", ".git", "requirements.txt", "setup.cfg")) -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return Path.cwd()

def resource_path(rel: str | Path) -> Path:
    rel = Path(rel)
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = project_root()
    return (base / rel).resolve()