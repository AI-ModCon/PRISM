"""vLLM integration for PRISM (image + text).

Usage:
    import src.vllm_plugin
    src.vllm_plugin.register()
    from vllm import LLM
    llm = LLM(model="<exported_prism_dir>", trust_remote_code=True)
"""

_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm import ModelRegistry
    from vllm.multimodal import MULTIMODAL_REGISTRY

    from .prism_for_conditional_generation import PrismForConditionalGeneration
    from .processor import (
        PrismDummyInputsBuilder,
        PrismMultiModalProcessor,
        PrismProcessingInfo,
    )

    ModelRegistry.register_model(
        "PrismForConditionalGeneration", PrismForConditionalGeneration
    )

    MULTIMODAL_REGISTRY.register_processor(
        PrismMultiModalProcessor,
        info=PrismProcessingInfo,
        dummy_inputs=PrismDummyInputsBuilder,
    )(PrismForConditionalGeneration)

    _REGISTERED = True


__all__ = ["register"]
