"""DistilBERT fine-tuning wrapper for binary phishing/legitimate classification.

Uses HuggingFace ``Trainer`` for a minimal, reproducible loop. Designed to run
on CPU or Apple Silicon MPS; for GPU just set ``CUDA_VISIBLE_DEVICES`` before
launch and Trainer picks it up.

The model exposes the same ``fit`` / ``predict_proba`` / ``save`` / ``load``
interface as :class:`nzphish.models.tfidf_lr.TfidfLR`, so it slots into
``nzphish.train`` and ``nzphish.eval.run`` without changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class DistilBertConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    seed: int = 42
    # Set True only if running on CUDA. MPS does not support fp16 reliably.
    fp16: bool = False
    # Truncate training set for fast smoke runs; -1 means use all.
    max_train_samples: int = -1


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DistilBertClassifier:
    """Thin wrapper around HuggingFace Trainer for binary text classification."""

    def __init__(self, cfg: DistilBertConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2
        )
        self.device = _select_device()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Featurization helpers
    # ------------------------------------------------------------------

    def _tokenize_fn(self, batch: dict) -> dict:
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.cfg.max_length,
        )

    def _to_dataset(self, texts: list[str], labels: np.ndarray | None) -> Dataset:
        data: dict = {"text": list(texts)}
        if labels is not None:
            data["label"] = [int(x) for x in labels]
        ds = Dataset.from_dict(data)
        return ds.map(self._tokenize_fn, batched=True, remove_columns=["text"])

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------

    def fit(self, texts: list[str], labels: np.ndarray) -> DistilBertClassifier:
        if self.cfg.max_train_samples > 0 and self.cfg.max_train_samples < len(texts):
            rng = np.random.default_rng(self.cfg.seed)
            idx = rng.choice(len(texts), size=self.cfg.max_train_samples, replace=False)
            texts = [texts[i] for i in idx]
            labels = labels[idx]

        train_ds = self._to_dataset(texts, labels)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        args = TrainingArguments(
            output_dir="results/_hf_tmp",
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            warmup_ratio=self.cfg.warmup_ratio,
            seed=self.cfg.seed,
            fp16=self.cfg.fp16 and self.device == "cuda",
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            processing_class=self.tokenizer,
            data_collator=collator,
        )
        trainer.train()
        return self

    def predict_proba(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        bs = batch_size or self.cfg.per_device_eval_batch_size
        ds = self._to_dataset(texts, labels=None)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model.eval()

        probs_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ds), bs):
                batch = ds[start : start + bs]
                features = [
                    {k: batch[k][i] for k in ("input_ids", "attention_mask")}
                    for i in range(len(batch["input_ids"]))
                ]
                padded = collator(features)
                padded = {k: v.to(self.device) for k, v in padded.items()}
                logits = self.model(**padded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_chunks.append(probs)
        return np.concatenate(probs_chunks, axis=0)

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "nzphish_distilbert_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def load(cls, path: Path) -> DistilBertClassifier:
        path = Path(path)
        cfg_dict = json.loads((path / "nzphish_distilbert_config.json").read_text())
        cfg = DistilBertConfig(**cfg_dict)
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.device = _select_device()
        obj.model.to(obj.device)
        return obj
