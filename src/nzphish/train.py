"""Training CLI: ``nzphish-train --config configs/tfidf_lr.yaml``.

Loads splits from ``data/processed/``, fits the configured model, evaluates on
val + test, writes metrics JSON to ``results/runs/{run_id}/``, and saves the
fitted model + a per-sample probability dump (used downstream by the deferral
layer for threshold calibration).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

from nzphish.eval.metrics import classification_metrics
from nzphish.models.tfidf_lr import TfidfLR, TfidfLRConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "runs"


def _build_model(name: str, params: dict):
    if name == "tfidf_lr":
        return TfidfLR(TfidfLRConfig(**params))
    raise ValueError(f"unknown model: {name}")


def _run_id(model_name: str) -> str:
    return f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"


@click.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--seed", default=42, show_default=True, type=int)
def main(config_path: str, seed: int) -> None:
    np.random.seed(seed)
    cfg = yaml.safe_load(Path(config_path).read_text())
    model_name = cfg["model"]
    params = cfg.get("params", {})

    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    click.echo(f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    model = _build_model(model_name, params)
    t0 = time.time()
    model.fit(train_df["text"].astype(str).tolist(), train_df["label"].to_numpy())
    fit_seconds = time.time() - t0
    click.echo(f"fit: {fit_seconds:.1f}s")

    out_dir = RESULTS_DIR / _run_id(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "config": cfg,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "splits": {},
    }
    for name, df in (("val", val_df), ("test", test_df)):
        prob = model.predict_proba(df["text"].astype(str).tolist())[:, 1]
        m = classification_metrics(df["label"].to_numpy(), prob)
        metrics["splits"][name] = m
        # Persist per-sample probs for deferral calibration
        out_probs = pd.DataFrame(
            {
                "id": df["id"].to_numpy(),
                "channel": df["channel"].to_numpy(),
                "label": df["label"].to_numpy(),
                "prob_phish": prob,
            }
        )
        out_probs.to_parquet(out_dir / f"{name}_probs.parquet", index=False)
        click.echo(f"{name}: f1_phish={m['f1_phish']:.4f} auc={m['roc_auc']:.4f}")

    model.save(out_dir / "model.pkl")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    click.echo(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
