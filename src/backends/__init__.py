"""Dependency-free backend metadata; importing this package never loads a model."""

from .capabilities import describe_backend, list_backends, resolve_backend
from .contracts import BackendDescriptor

__all__ = ("BackendDescriptor", "list_backends", "describe_backend", "resolve_backend")
