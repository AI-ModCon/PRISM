import math

from torch.optim.lr_scheduler import LambdaLR


def get_molmo_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_connector: int = 200,
    warmup_main: int = 2000,
    min_lr_ratio: float = 0.1,
):
    """
    Creates a detailed LambdaLR scheduler matching Molmo pre-training recipe.

    Assumptions on Optimizer Param Groups:
    - Group 0: Connector (High LR, Short Warmup)
    - Group 1: ViT (Low LR, Long Warmup)
    - Group 2: LLM (Med LR, Long Warmup)

    If groups don't match exactly, fallback logic is applied based on index.
    """

    def lr_lambda_connector(current_step: int):
        # Warmup
        if current_step < warmup_connector:
            return float(current_step) / float(max(1, warmup_connector))

        # Cosine Decay to min_lr_ratio
        progress = float(current_step - warmup_connector) / float(
            max(1, num_training_steps - warmup_connector)
        )
        progress = min(max(progress, 0.0), 1.0)  # Clip 0-1

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    def lr_lambda_main(current_step: int):
        # Warmup
        if current_step < warmup_main:
            return float(current_step) / float(max(1, warmup_main))

        # Cosine Decay
        progress = float(current_step - warmup_main) / float(
            max(1, num_training_steps - warmup_main)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    # Create list of lambdas matching the number of parameter groups
    # We assume the Trainer sets up groups in this order: [Connector, ViT, LLM]
    # If there are more or fewer groups, we need to be robust.

    num_groups = len(optimizer.param_groups)
    lr_lambdas = []

    for i in range(num_groups):
        if i == 0:
            # Connector (Always first in our custom group logic)
            lr_lambdas.append(lr_lambda_connector)
        else:
            # ViT, LLM, etc.
            lr_lambdas.append(lr_lambda_main)

    return LambdaLR(optimizer, lr_lambdas)


def get_cosine_with_min_lr(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1
):
    """
    Standard Cosine with Warmup but decaying to a floor (min_lr_ratio) instead of 0.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_wsd_scheduler(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    decay_ratio: float = 0.1,
    decay_steps: int | None = None,
):
    """Warmup-stable-decay scheduler.

    The LR linearly warms up to the base LR, stays flat, then optionally
    cosine-decays to ``min_lr_ratio`` over the final decay window. Set
    ``decay_steps=0`` or ``decay_ratio=0`` for warmup-stable only.
    """
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    no_decay = decay_steps == 0 or (decay_steps is None and decay_ratio == 0)
    if decay_steps is None:
        if not 0.0 <= decay_ratio <= 1.0:
            raise ValueError("decay_ratio must be in [0, 1]")
        decay_steps = max(1, int(round(num_training_steps * decay_ratio)))
    if decay_steps < 0:
        raise ValueError("decay_steps must be non-negative")
    if decay_steps == 0:
        no_decay = True

    warmup_steps = min(num_warmup_steps, num_training_steps)

    if no_decay:
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    # By here decay_steps is guaranteed >= 1: the None and <= 0 cases were
    # resolved into no_decay (handled above) or rejected earlier.
    decay_steps = min(decay_steps, num_training_steps)
    decay_start = max(warmup_steps, num_training_steps - decay_steps)
    actual_decay_steps = max(1, num_training_steps - decay_start)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        if current_step < decay_start:
            return 1.0

        progress = float(current_step - decay_start) / float(actual_decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
