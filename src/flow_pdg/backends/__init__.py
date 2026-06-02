"""Backend registry / factory."""

from __future__ import annotations

from .base import Conditioning, KVCache, VelocityBackend, VelocityOutput


def make_backend(name: str, **kwargs) -> VelocityBackend:
    if name == "dummy":
        from .dummy import DummyBackend
        return DummyBackend(**kwargs)
    if name == "sd3":
        from .sd3 import SD3Backend
        return SD3Backend(**kwargs)
    if name == "sd3_depth":
        from .sd3_depth import SD3DepthBackend
        return SD3DepthBackend(**kwargs)
    if name == "flux":
        from .flux import FluxBackend
        return FluxBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r}")


__all__ = [
    "Conditioning",
    "KVCache",
    "VelocityBackend",
    "VelocityOutput",
    "make_backend",
]
