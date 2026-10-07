import io
import json
from bisect import bisect_right
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .image_transforms import build_image_transform


class CalvinVLADataset(Dataset):
    """CALVIN VLA samples: {top image, wrist image, pose, text} -> next action."""

    def __init__(
        self,
        root_dir,
        tokenizer,
        split="train",
        max_chunk=20,
        text_max_length=128,
        cache_size=8,
        strict_integrity=False,
        max_skipped_fraction=1.0,
        model_config=None,
    ):
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_chunk = int(max_chunk)
        self.text_max_length = int(text_max_length)
        self.cache_size = int(cache_size)
        self.strict_integrity = bool(strict_integrity)
        self.max_skipped_fraction = float(max_skipped_fraction)

        if self.max_chunk > 20:
            raise ValueError(f"max_chunk must be <= 20, got {self.max_chunk}")
        if not (0.0 <= self.max_skipped_fraction <= 1.0):
            raise ValueError(
                f"max_skipped_fraction must be in [0, 1], got {self.max_skipped_fraction}"
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        self.info = json.loads((self.root_dir / "meta" / "info.json").read_text())
        split_range = self.info["splits"].get(split)
        if split_range is None:
            raise ValueError(
                f"Unknown split '{split}'. Available: {list(self.info['splits'].keys())}"
            )
        split_start, split_end = [int(x) for x in split_range.split(":")]

        self.task_by_index = {}
        with open(self.root_dir / "meta" / "tasks.jsonl") as f:
            for line in f:
                row = json.loads(line)
                self.task_by_index[int(row["task_index"])] = row["task"]

        self.episode_task_fallback = {}
        episode_length = {}
        with open(self.root_dir / "meta" / "episodes.jsonl") as f:
            for line in f:
                row = json.loads(line)
                ep_idx = int(row["episode_index"])
                episode_length[ep_idx] = int(row["length"])
                tasks = row.get("tasks", [])
                if tasks:
                    self.episode_task_fallback[ep_idx] = tasks[0]

        self.episode_ids = []
        self.steps_per_episode = []
        total_split_episodes = split_end - split_start
        skipped_chunk = 0
        skipped_missing_parquet = 0
        skipped_short = 0
        considered_episodes = 0
        for episode_idx in range(split_start, split_end):
            if episode_idx // 1000 > self.max_chunk:
                skipped_chunk += 1
                continue
            considered_episodes += 1
            parquet_path = self._episode_path(episode_idx)
            if not parquet_path.exists():
                skipped_missing_parquet += 1
                continue
            length = int(episode_length.get(episode_idx, 0))
            if length < 2:
                skipped_short += 1
                continue
            self.episode_ids.append(episode_idx)
            self.steps_per_episode.append(length - 1)

        skipped_considered = skipped_missing_parquet + skipped_short
        self.dataset_stats = {
            "split_total_episodes": total_split_episodes,
            "considered_episodes": considered_episodes,
            "kept_episodes": len(self.episode_ids),
            "skipped_chunk": skipped_chunk,
            "skipped_missing_parquet": skipped_missing_parquet,
            "skipped_short": skipped_short,
        }
        if skipped_considered > 0:
            print(
                "[calvin_vla] Episode filtering summary: "
                f"kept={len(self.episode_ids)} "
                f"skipped_missing_parquet={skipped_missing_parquet} "
                f"skipped_short={skipped_short} "
                f"skipped_chunk={skipped_chunk}"
            )
        if self.strict_integrity and skipped_considered > 0:
            raise RuntimeError(
                "strict_integrity=True and CALVIN episodes were skipped: "
                f"missing_parquet={skipped_missing_parquet}, short={skipped_short}"
            )
        if considered_episodes > 0:
            skipped_fraction = skipped_considered / considered_episodes
            if skipped_fraction > self.max_skipped_fraction:
                raise RuntimeError(
                    f"CALVIN skipped fraction {skipped_fraction:.3f} exceeds "
                    f"max_skipped_fraction={self.max_skipped_fraction:.3f}"
                )

        if not self.episode_ids:
            raise RuntimeError(
                f"No valid CALVIN episodes found for split={split}, max_chunk<={self.max_chunk}"
            )

        self.cumulative_steps = []
        running = 0
        for steps in self.steps_per_episode:
            running += steps
            self.cumulative_steps.append(running)
        self.total_steps = running

        self.image_transform = build_image_transform(model_config)
        self._episode_cache = {}
        self._cache_order = []
        self._token_cache = {}
        self.active_modalities = {"image", "pose", "text"}

    def _episode_path(self, episode_idx):
        chunk = episode_idx // 1000
        return self.root_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"

    def _load_episode(self, episode_idx):
        if episode_idx in self._episode_cache:
            return self._episode_cache[episode_idx]

        if episode_idx // 1000 > self.max_chunk:
            raise RuntimeError(f"Episode {episode_idx} resolves to chunk>{self.max_chunk}")

        path = self._episode_path(episode_idx)
        df = pd.read_parquet(
            path,
            columns=[
                "action",
                "observation.state",
                "observation.images.top",
                "observation.images.wrist",
                "task_index",
                "frame_index",
                "episode_index",
            ],
        )

        self._episode_cache[episode_idx] = df
        self._cache_order.append(episode_idx)
        if len(self._cache_order) > self.cache_size:
            old_key = self._cache_order.pop(0)
            self._episode_cache.pop(old_key, None)
        return df

    def _tokenize_text(self, text):
        if text not in self._token_cache:
            token_ids = self.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.text_max_length,
            ).input_ids.squeeze(0)
            if token_ids.numel() == 0:
                raise RuntimeError("Tokenizer returned empty sequence for CALVIN instruction")
            self._token_cache[text] = token_ids
        return self._token_cache[text].clone()

    def _decode_image(self, image_entry):
        if not isinstance(image_entry, dict) or "bytes" not in image_entry:
            raise RuntimeError("Expected image entry with embedded PNG bytes")
        image = Image.open(io.BytesIO(image_entry["bytes"])).convert("RGB")
        return self.image_transform(image)

    def __len__(self):
        return self.total_steps

    def __getitem__(self, index):
        if index < 0 or index >= self.total_steps:
            raise IndexError(f"Index out of range: {index}")

        episode_pos = bisect_right(self.cumulative_steps, index)
        prev_steps = 0 if episode_pos == 0 else self.cumulative_steps[episode_pos - 1]
        step_idx = index - prev_steps
        episode_idx = self.episode_ids[episode_pos]

        episode_df = self._load_episode(episode_idx)
        curr = episode_df.iloc[step_idx]
        nxt = episode_df.iloc[step_idx + 1]

        task_idx = int(curr["task_index"])
        text = self.task_by_index.get(task_idx) or self.episode_task_fallback.get(episode_idx)
        if not text:
            raise RuntimeError(
                f"Missing instruction for episode={episode_idx}, task_index={task_idx}"
            )

        pose = torch.tensor(curr["observation.state"], dtype=torch.float32)
        action = torch.tensor(nxt["action"], dtype=torch.float32)
        if pose.ndim != 1:
            raise RuntimeError(f"Pose must be rank-1, got shape {tuple(pose.shape)}")
        if action.ndim != 1:
            raise RuntimeError(f"Action must be rank-1, got shape {tuple(action.shape)}")

        text_ids = self._tokenize_text(text)
        text_mask = torch.ones_like(text_ids, dtype=torch.long)

        return {
            "image_head": self._decode_image(curr["observation.images.top"]),
            "image_wrist": self._decode_image(curr["observation.images.wrist"]),
            "pose": pose,
            "action": action,
            "text": text_ids,
            "text_attention_mask": text_mask,
            "_metadata": f"[calvin] ep={episode_idx} step={step_idx} task={task_idx}",
        }
