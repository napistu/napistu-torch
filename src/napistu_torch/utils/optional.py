"""Utilities for handling optional dependencies in napistu-torch."""

from __future__ import annotations

import importlib
import importlib.util
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from napistu_torch.constants import OPTIONAL_DEFS

_F = TypeVar("_F", bound=Callable[..., Any])


def is_lightning_available() -> bool:
    """Return True if pytorch_lightning can be imported."""

    return importlib.util.find_spec(OPTIONAL_DEFS.LIGHTNING_PACKAGE) is not None


def import_lightning():
    """Import and return pytorch_lightning, raising an informative error if missing."""

    try:
        return importlib.import_module(OPTIONAL_DEFS.LIGHTNING_PACKAGE)
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - executed when dependency missing
        raise ImportError(
            "This functionality requires `pytorch_lightning`. "
            f"Install with `pip install napistu-torch[{OPTIONAL_DEFS.LIGHTNING_EXTRA}]`."
        ) from exc


def require_lightning(func: _F) -> _F:
    """Decorator ensuring pytorch_lightning is available before calling *func*."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_lightning()
        return func(*args, **kwargs)

    return cast(_F, wrapper)
