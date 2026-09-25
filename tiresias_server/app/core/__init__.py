from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

__all__ = ["Engine"]


def __getattr__(name: str):
    if name == "Engine":
        from .engine import Engine
        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
