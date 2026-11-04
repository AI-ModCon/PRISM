"""Stable backend discovery and metadata-only resolution; no execution imports."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from .contracts import BackendDescriptor, _capability_set

_KNOWN_CAPABILITIES = frozenset({"causal_forward", "autoregressive_generate"})

# No mutable reference to the underlying registry is exposed. Dict insertion
# order is the public discovery order for contract version 1.
_BACKENDS: Mapping[str, BackendDescriptor] = MappingProxyType(
    {
        "local_ar": BackendDescriptor("local_ar", capabilities=frozenset({"causal_forward"})),
        "hf_ar": BackendDescriptor("hf_ar", capabilities=_KNOWN_CAPABILITIES),
        "u2_reference": BackendDescriptor("u2_reference", status="planned"),
        "u2_native": BackendDescriptor("u2_native", status="planned"),
    }
)


def list_backends() -> tuple[str, ...]:
    """Return all builtin names in stable order, including planned backends."""
    return tuple(_BACKENDS)


def describe_backend(name: str) -> BackendDescriptor:
    """Return immutable metadata for an exact builtin name, even when planned.

    Raises ValueError for malformed or unknown names. No model is loaded.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("backend name must be a nonempty string")
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; available backends: {', '.join(_BACKENDS)}")
    return _BACKENDS[name]


def resolve_backend(
    *,
    backend: str | None = None,
    llm_backbone_id: str | None = None,
    contract_version: int = 1,
    required_capabilities: Iterable[str] = (),
) -> BackendDescriptor:
    """Validate selectors and return metadata for an existing backend.

    With no explicit backend, a nonempty ``llm_backbone_id`` selects ``hf_ar``;
    None or an empty string selects ``local_ar``. Explicit selectors must agree
    with that rule. The legacy weight-initialization field ``hf_model_id`` is
    intentionally not a selector or argument here.

    Only integer contract version 1 is supported. Requested capabilities must
    be an iterable of known names (not a bare string or mapping). Invalid input
    or an unsupported capability raises ValueError. Planned backends raise
    NotImplementedError after input validation, with no fallback.

    This does not construct a model, validate a model ID against a provider,
    check installed dependencies, or alter runtime routing.
    """
    if type(contract_version) is not int or contract_version != 1:
        raise ValueError(f"Unsupported contract_version {contract_version!r}; expected integer 1")
    if llm_backbone_id is not None and not isinstance(llm_backbone_id, str):
        raise ValueError("llm_backbone_id must be a string or None")

    requested = _capability_set(required_capabilities, field="required_capabilities")
    unknown = requested - _KNOWN_CAPABILITIES
    if unknown:
        raise ValueError(
            f"Unknown required_capabilities: {', '.join(sorted(unknown))}; "
            f"known capabilities: {', '.join(sorted(_KNOWN_CAPABILITIES))}"
        )

    selected = backend if backend is not None else ("hf_ar" if llm_backbone_id else "local_ar")
    descriptor = describe_backend(selected)
    if descriptor.status == "planned":
        raise NotImplementedError(f"Backend {selected!r} is planned and unavailable; no fallback")
    if selected == "hf_ar" and not llm_backbone_id:
        raise ValueError("backend 'hf_ar' requires a nonempty llm_backbone_id")
    if selected == "local_ar" and llm_backbone_id:
        raise ValueError(
            "backend 'local_ar' conflicts with a nonempty llm_backbone_id (selects 'hf_ar')"
        )

    unsupported = requested - descriptor.capabilities
    if unsupported:
        raise ValueError(
            f"Backend {selected!r} does not support required_capabilities: "
            f"{', '.join(sorted(unsupported))}"
        )
    return descriptor
