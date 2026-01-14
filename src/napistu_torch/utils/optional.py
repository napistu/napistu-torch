"""
Utilities for handling optional dependencies in Napistu-Torch.

Decorators
----------
require_bionty
    Decorator ensuring bionty is available before calling *func*.
require_lightning
    Decorator ensuring pytorch_lightning is available before calling *func*.
require_modelgenerator
    Decorator ensuring modelgenerator is available before calling *func*.
require_scdataloader
    Decorator ensuring scdataloader is available before calling *func*.
require_scgpt
    Decorator ensuring scgpt is available before calling *func*.
require_scprint
    Decorator ensuring scprint is available before calling *func*.
require_seaborn
    Decorator ensuring seaborn is available before calling *func*.
require_torchtext
    Decorator ensuring torchtext is available before calling *func*.

Public Functions
----------------
import_bionty:
    Import and return bionty, raising an informative error if missing.
import_lightning:
    Import and return pytorch_lightning, raising an informative error if missing.
import_modelgenerator:
    Import and return modelgenerator, raising an informative error if missing.
import_scdataloader:
    Import and return scdataloader, raising an informative error if missing.
import_scgpt:
    Import and return scgpt, raising an informative error if missing.
import_scprint:
    Import and return scprint, raising an informative error if missing.
import_seaborn
    Import and return seaborn, raising an informative error if missing.
import_torchtext:
    Import and return torchtext, raising an informative error if missing.
"""

from __future__ import annotations

import importlib
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from napistu_torch.constants import OPTIONAL_DEFS

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def import_bionty():
    """Import and return bionty, raising an informative error if missing."""

    try:
        return importlib.import_module("bionty")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "This functionality requires `bionty`. "
            "Install with `pip install bionty` or `pip install lamindb[bionty]`."
        ) from exc


def require_bionty(func: _F) -> _F:
    """Decorator ensuring bionty is available before calling *func*.

    Use this decorator for scPRINT functions that require bionty/lamin.

    Examples
    --------
    >>> @require_bionty
    >>> def populate_lamin_db():
    ...     # Uses bionty
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_bionty()
        return func(*args, **kwargs)

    return cast(_F, wrapper)


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


def import_modelgenerator():
    """Import and return modelgenerator, raising an informative error if missing."""

    try:
        modelgenerator = importlib.import_module("modelgenerator")
        # Check version and warn if not 0.1.2
        try:
            # Try to get version from __version__ attribute first
            version = getattr(modelgenerator, "__version__", None)
            # If not found, use importlib.metadata (Python 3.11+)
            if version is None:
                from importlib.metadata import version as get_package_version

                version = get_package_version("modelgenerator")

            if version != "0.1.2":
                logger.warning(
                    f"Expected modelgenerator==0.1.2, but found version {version}. "
                    "This may cause compatibility issues. Install with `pip install modelgenerator==0.1.2`."
                )
        except Exception:
            # If version check fails, continue anyway
            pass
        return modelgenerator
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "This functionality requires `modelgenerator`. "
            "Install with `pip install modelgenerator==0.1.2`."
        ) from exc


def require_modelgenerator(func: _F) -> _F:
    """Decorator ensuring modelgenerator is available before calling *func*.

    Use this decorator for AIDOCell and scFoundation-specific functions.

    Examples
    --------
    >>> @require_modelgenerator
    >>> def load_aidocell_model(model_class):
    ...     # Uses modelgenerator
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_modelgenerator()
        return func(*args, **kwargs)

    return cast(_F, wrapper)


def import_scdataloader():
    """Import and return scdataloader, raising an informative error if missing."""

    try:
        return importlib.import_module("scdataloader")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "This functionality requires `scdataloader`. "
            "Install with `pip install scdataloader` (typically bundled with scprint)."
        ) from exc


def require_scdataloader(func: _F) -> _F:
    """Decorator ensuring scdataloader is available before calling *func*.

    Use this decorator for scPRINT functions that require scdataloader.

    Examples
    --------
    >>> @require_scdataloader
    >>> def populate_lamin_db():
    ...     # Uses scdataloader.utils.populate_my_ontology
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_scdataloader()
        return func(*args, **kwargs)

    return cast(_F, wrapper)


def import_scgpt():
    """Import and return scgpt, raising an informative error if missing."""

    try:
        return importlib.import_module("scgpt")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "This functionality requires `scgpt`. "
            "Install with `pip install scgpt wandb gseapy`. "
            "If you encounter issues with `torchtext` compatibility (common with PyTorch), "
            "use conda/mamba instead: "
            "`mamba install pytorch torchtext==0.18.0 -c pytorch -c conda-forge` "
            "then `pip install scgpt wandb gseapy`."
        ) from exc


def require_scgpt(func: _F) -> _F:
    """Decorator ensuring scgpt is available before calling *func*.

    Use this decorator for scGPT-specific functions.

    Examples
    --------
    >>> @require_scgpt
    >>> def load_scgpt_model(model_dir):
    ...     # Uses scgpt
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_scgpt()
        return func(*args, **kwargs)

    return cast(_F, wrapper)


def import_scprint():
    """Import and return scprint, raising an informative error if missing."""

    try:
        return importlib.import_module("scprint")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "This functionality requires `scprint`. "
            "Install with `pip install scprint`."
        ) from exc


def require_scprint(func: _F) -> _F:
    """Decorator ensuring scprint is available before calling *func*.

    Use this decorator for scPRINT-specific functions.

    Examples
    --------
    >>> @require_scprint
    >>> def load_scprint_model(checkpoint_path):
    ...     # Uses scprint
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_scprint()
        return func(*args, **kwargs)

    return cast(_F, wrapper)


def import_seaborn():
    """Import and return seaborn, raising an informative error if missing."""

    try:
        return importlib.import_module(OPTIONAL_DEFS.SEABORN_PACKAGE)
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - executed when dependency missing
        raise ImportError(
            "This functionality requires `seaborn`. "
            f"Install with `pip install napistu-torch[{OPTIONAL_DEFS.SEABORN_EXTRA}]`."
        ) from exc


def require_seaborn(func: _F) -> _F:
    """Decorator ensuring seaborn is available before calling *func*."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_seaborn()
        return func(*args, **kwargs)

    return cast(_F, wrapper)


def import_torchtext():
    """Import and return torchtext, raising an informative error if missing."""

    try:
        return importlib.import_module("torchtext")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "This functionality requires `torchtext`. "
            "Install with `pip install torchtext==0.18.0`. "
            "If you encounter compatibility issues with PyTorch, "
            "use conda/mamba instead: "
            "`mamba install pytorch torchtext==0.18.0 -c pytorch -c conda-forge`."
        ) from exc


def require_torchtext(func: _F) -> _F:
    """Decorator ensuring torchtext is available before calling *func*.

    Use this decorator for scGPT functions that require torchtext.

    Examples
    --------
    >>> @require_torchtext
    >>> def load_scgpt(model_dir):
    ...     # Uses torchtext.vocab.Vocab
    ...     pass
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        import_torchtext()
        return func(*args, **kwargs)

    return cast(_F, wrapper)
