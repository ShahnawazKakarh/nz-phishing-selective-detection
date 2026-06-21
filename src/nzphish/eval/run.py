"""Stratified evaluation: per-source and per-channel metrics.

The aggregate test-set number hides the question we actually care about:
*do NZ-localized features help on NZ-specific samples?* This script loads the
per-sample probability dump written by ``nzphish-train`` and computes
classification metrics over the (source, channel) subgroups.

Usage:
    nzphish-eval-stratified --run-dir results/runs/tfidf_lr_20260621_232945
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from nzphish.eval.metrics import classification_metrics

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "processed"


def _join_source(probs: pd.DataFrame, split: str) -> pd.DataFrame:
    splits = pd.read_parquet(DATA_DIR / f"{split}.parquet")[["id", "source"]]
    return probs.merge(splits, on="id", how="left")


@click.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--split", default="test", show_default=True, type=click.Choice(["val", "test"]))
def main(run_dir: str, split: str) -> None:
    run_path = Path(run_dir)
    probs = pd.read_parquet(run_path / f"{split}_probs.parquet")
    df = _join_source(probs, split)

    rows: list[dict] = []

    # Overall
    m = classification_metrics(df["label"].to_numpy(), df["prob_phish"].to_numpy())
    rows.append(
        {
            "subgroup": "overall",
            "n": m["n"],
            **{
                k: m[k]
                for k in ("accuracy", "precision_phish", "recall_phish", "f1_phish", "roc_auc")
            },
        }
    )

    # By source
    for source, sub in df.groupby("source"):
        if len(set(sub["label"])) < 2:
            continue
        m = classification_metrics(sub["label"].to_numpy(), sub["prob_phish"].to_numpy())
        rows.append(
            {
                "subgroup": f"source={source}",
                "n": m["n"],
                **{
                    k: m[k]
                    for k in ("accuracy", "precision_phish", "recall_phish", "f1_phish", "roc_auc")
                },
            }
        )

    # By channel
    for channel, sub in df.groupby("channel"):
        if len(set(sub["label"])) < 2:
            continue
        m = classification_metrics(sub["label"].to_numpy(), sub["prob_phish"].to_numpy())
        rows.append(
            {
                "subgroup": f"channel={channel}",
                "n": m["n"],
                **{
                    k: m[k]
                    for k in ("accuracy", "precision_phish", "recall_phish", "f1_phish", "roc_auc")
                },
            }
        )

    out = pd.DataFrame(rows)
    click.echo(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out_path = run_path / f"{split}_stratified.json"
    out_path.write_text(json.dumps(rows, indent=2))
    click.echo(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
