"""Immutable, dependency-free metadata for the backend discovery contract."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal


def _capability_set(values: Iterable[str], *, field: str) -> frozenset[str]:
    """Copy an iterable of capability names without coercing malformed inputs."""
    if isinstance(values, str | bytes | bytearray | Mapping):
        raise ValueError(
            f"{field} must be an iterable of capability names, not a string or mapping"
        )
    try:
        items = iter(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be an iterable of capability names") from exc
    names = set()
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field} must contain nonempty strings; got {item!r}")
        names.add(item)
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    """Backend metadata, never a model or a guarantee that assets are available.

    ``capabilities`` is copied to a frozenset to prevent caller-owned mutable
    collections from changing a descriptor. ``status`` distinguishes current
    execution paths from reserved, unavailable backend names.
    """

    name: str
    contract_version: int = 1
    capabilities: frozenset[str] = frozenset()
    status: Literal["existing", "planned"] = "existing"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capabilities", _capability_set(self.capabilities, field="capabilities")
        )
