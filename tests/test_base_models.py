import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.slow,
]

OLMO_BACKBONES = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-7B-0724-hf",
]


@pytest.mark.parametrize("model_id", OLMO_BACKBONES)
def test_base_model_generation_smoke(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert isinstance(output_text, str)
    assert len(output_text) > 0
