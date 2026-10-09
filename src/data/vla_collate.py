import torch
from torch.nn.utils.rnn import pad_sequence


class VLACollator:
    """Pads text fields and stacks fixed-shape VLA tensors."""

    def __init__(self, tokenizer):
        self.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    def __call__(self, batch):
        text = [item["text"] for item in batch]
        text_mask = [item["text_attention_mask"] for item in batch]

        return {
            "text": pad_sequence(text, batch_first=True, padding_value=self.pad_token_id),
            "text_attention_mask": pad_sequence(text_mask, batch_first=True, padding_value=0),
            "image_head": torch.stack([item["image_head"] for item in batch]),
            "image_wrist": torch.stack([item["image_wrist"] for item in batch]),
            "pose": torch.stack([item["pose"] for item in batch]),
            "action": torch.stack([item["action"] for item in batch]),
            "_metadata": [item["_metadata"] for item in batch],
        }
