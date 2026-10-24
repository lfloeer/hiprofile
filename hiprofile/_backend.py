"""Array backend selection (NumPy or JAX)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

BACKENDS = ("numpy", "jax")


def has_jax() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def get_xp(backend: str) -> ModuleType:
    """Return the array namespace for ``backend``.

    The JAX backend always runs in double precision; the model is compared to the
    golden files at the 1e-6 level and the marginal likelihood involves large
    cancellations.
    """
    if backend == "numpy":
        return np
    if backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        return jnp
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")


def default_backend() -> str:
    return "numpy"


def to_numpy(a: Any) -> np.ndarray:
    return np.asarray(a)
