"""Pluggable LLM backends shared by stage 03 (cross-DB ID resolution
fallback) and stage 09 (question/answer/reasoning generation).

Two backends:
- ClaudeBackend  — real generation via the Anthropic Messages API.
- MockBackend    — zero-cost, deterministic placeholder output, for
                   validating the rest of the pipeline without API cost.

Selected via CONFIG.reasoning_backend ("claude" | "mock").
"""

import json
import logging
from abc import ABC, abstractmethod

from src.data.kegg_curation.config import CONFIG

logger = logging.getLogger(__name__)


class ReasoningBackend(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Single-turn text completion. Returns the raw response text."""

    @abstractmethod
    def generate_variant_reasoning(self, variant_context: dict) -> dict:
        """Structured generation for stage 09: given a variant's context
        (gene, network/pathway, disease, coordinates, etc.), returns
        {"question": ..., "answer": ..., "reasoning": ...}.
        """


class ClaudeBackend(ReasoningBackend):
    def __init__(self) -> None:
        if not CONFIG.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or add it to .env "
                "before using reasoning_backend='claude'."
            )
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
        # Some models (confirmed: claude-sonnet-5) reject an explicit
        # `temperature` as a deprecated parameter. Learned once on first
        # call and remembered for the rest of this backend instance's
        # lifetime, so a run of hundreds of calls doesn't pay for a wasted
        # 400 on every single one.
        self._temperature_supported = True

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        kwargs: dict = {
            "model": CONFIG.anthropic_model,
            "max_tokens": CONFIG.reasoning_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self._temperature_supported:
            kwargs["temperature"] = CONFIG.reasoning_temperature
        try:
            response = self._client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as e:
            if self._temperature_supported and "temperature" in str(e) and "deprecated" in str(e):
                self._temperature_supported = False
                kwargs.pop("temperature", None)
                response = self._client.messages.create(**kwargs)
            else:
                raise
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    def generate_variant_reasoning(self, variant_context: dict) -> dict:
        system = (
            "You are a genetics expert analyzing disease-causing mutations. "
            "Provide your analysis in VALID JSON format only, with no "
            "markdown formatting or explanatory text. Your JSON must contain "
            "exactly these keys: question, answer, reasoning."
        )
        prompt = (
            "Analyze this genetic variant and its biological/disease "
            "significance:\n\n" + json.dumps(variant_context, indent=2)
            + "\n\nRespond with JSON: "
            '{"question": "...", "answer": "concise 2-3 sentence disease '
            'mechanism summary", "reasoning": "step-by-step molecular -> '
            'protein -> pathway -> disease reasoning"}'
        )
        raw = self.complete(prompt, system=system)
        return _parse_json_response(raw)


class MockBackend(ReasoningBackend):
    """Zero-cost, deterministic placeholder — for dry-running the pipeline
    mechanics (stages 01-10 end to end) without any API cost.
    """

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return "[]"

    def generate_variant_reasoning(self, variant_context: dict) -> dict:
        gene = variant_context.get("gene", "UNKNOWN_GENE")
        disease = variant_context.get("disease", "an unspecified disease")
        return {
            "question": (
                f"[MOCK] What is the biological effect of this {gene} "
                f"variant, and what disease does it contribute to?"
            ),
            "answer": f"[MOCK] This variant in {gene} is associated with {disease}.",
            "reasoning": (
                f"[MOCK] Step 1: variant affects {gene}. "
                f"Step 2: disrupted protein function. "
                f"Step 3: pathway dysregulation. "
                f"Step 4: contributes to {disease}."
            ),
        }


def _parse_json_response(raw: str) -> dict:
    """Multi-tier JSON extraction, mirroring the original notebook's
    robustness chain: direct parse -> strip markdown fences -> bracket-match.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        try:
            return json.loads(stripped.strip())
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON from LLM response: {raw[:200]!r}")
    return {"error": "unparseable_response", "raw_response": raw}


_backend_cache: ReasoningBackend | None = None


def get_reasoning_backend() -> ReasoningBackend:
    global _backend_cache
    if _backend_cache is not None:
        return _backend_cache

    if CONFIG.reasoning_backend == "claude":
        _backend_cache = ClaudeBackend()
    elif CONFIG.reasoning_backend == "mock":
        _backend_cache = MockBackend()
    else:
        raise ValueError(
            f"Unknown reasoning_backend: {CONFIG.reasoning_backend!r} "
            "(expected 'claude' or 'mock')"
        )
    return _backend_cache
