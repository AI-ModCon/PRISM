"""Training package exports with lazy imports to avoid eager heavy dependencies."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer_native import train_native_ddp
    from .trainer_zone_a import ZoneATrainer

from .distributed import (
    load_model_weights_only,
    save_native_ddp_checkpoint,
    setup_distributed,
    wrap_model_distributed,
)

__all__ = [
    "ZoneATrainer",
    "train_native_ddp",
    "setup_distributed",
    "wrap_model_distributed",
    "save_native_ddp_checkpoint",
    "load_model_weights_only",
]


def __getattr__(name):
    if name == "ZoneATrainer":
        from .trainer_zone_a import ZoneATrainer

        return ZoneATrainer
    if name == "train_native_ddp":
        from .trainer_native import train_native_ddp

        return train_native_ddp

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
