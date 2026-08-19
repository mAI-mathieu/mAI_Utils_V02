"""Optional loader for the installed ComfyUI-Inpaint-CropAndStitch node pack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class CropAndStitchDependencyError(ImportError):
    """Raised when the optional CropAndStitch dependency cannot be used."""

    def __init__(self, message: str, *, missing: bool = False):
        super().__init__(message)
        self.missing = missing


_PRIVATE_MODULE_NAME = "_mai_inpaint_cropandstitch_dependency"
_REQUIRED_ATTRIBUTES = (
    "InpaintCropImproved",
    "InpaintStitchImproved",
    "CPUProcessorLogic",
    "GPUProcessorLogic",
)


def _is_core_module(module: object) -> bool:
    return all(hasattr(module, name) for name in _REQUIRED_ATTRIBUTES)


def _already_loaded_module() -> ModuleType | None:
    private_module = sys.modules.get(_PRIVATE_MODULE_NAME)
    if private_module is not None and _is_core_module(private_module):
        return private_module

    for module in tuple(sys.modules.values()):
        if module is not None and _is_core_module(module):
            module_file = str(getattr(module, "__file__", "")).lower()
            if "inpaint_cropandstitch" in module_file:
                return module
    return None


def _dependency_candidates() -> list[Path]:
    custom_nodes_dir = Path(__file__).resolve().parents[2]
    candidates = []
    for child in custom_nodes_dir.iterdir():
        if not child.is_dir():
            continue
        core_file = child / "inpaint_cropandstitch.py"
        if core_file.is_file():
            candidates.append(core_file)

    return sorted(
        candidates,
        key=lambda path: (
            "inpaint-cropandstitch" not in path.parent.name.lower(),
            path.parent.name.lower(),
        ),
    )


def load_cropandstitch_module() -> ModuleType:
    """Return the dependency's core module without assuming a Python-safe folder name."""

    loaded = _already_loaded_module()
    if loaded is not None:
        return loaded

    candidates = _dependency_candidates()
    if not candidates:
        raise CropAndStitchDependencyError(
            "ComfyUI-Inpaint-CropAndStitch is not installed.", missing=True
        )

    core_file = candidates[0]
    spec = importlib.util.spec_from_file_location(_PRIVATE_MODULE_NAME, core_file)
    if spec is None or spec.loader is None:
        raise CropAndStitchDependencyError(
            f"Could not create an import specification for {core_file.parent.name}."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PRIVATE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(_PRIVATE_MODULE_NAME, None)
        raise CropAndStitchDependencyError(
            f"Failed to import {core_file.parent.name}: {exc}"
        ) from exc

    if not _is_core_module(module):
        sys.modules.pop(_PRIVATE_MODULE_NAME, None)
        raise CropAndStitchDependencyError(
            f"{core_file.parent.name} does not expose the expected crop/stitch API."
        )
    return module
