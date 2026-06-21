"""Collect the UCI SMS Spam Collection dataset.

5,574 SMS messages labeled ham/spam. This is the global SMS-spam baseline.
It is *not* NZ-localized, but provides class balance for the SMS channel and
exposes the model to standard smishing patterns before we layer NZ-specific
synthetic lures on top.

Source: UCI ML Repository (Almeida & Hidalgo, 2011) via the HuggingFace mirror.
"""

from __future__ import annotations

from pathlib import Path

import click
from datasets import load_dataset

from nzphish.data.schema import CANONICAL_COLUMNS, CHANNEL_SMS, LABEL_LEGIT, LABEL_PHISH

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "processed" / "sources" / "uci_sms.parquet"


@click.command()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ucirvine/sms_spam", split="train")
    df = ds.to_pandas()
    # HF schema: 'sms' (text), 'label' (0=ham, 1=spam) -- treat spam as phish-positive
    df = df.rename(columns={"sms": "text"})
    df["channel"] = CHANNEL_SMS
    df["source"] = "uci_sms"
    df["lang"] = "en"
    df["label"] = df["label"].astype(int)
    assert set(df["label"].unique()) <= {LABEL_LEGIT, LABEL_PHISH}
    df["id"] = [f"uci_sms_{i}" for i in range(len(df))]
    df = df[list(CANONICAL_COLUMNS)]
    df.to_parquet(OUT, index=False)
    click.echo(f"wrote {OUT}: {len(df):,} rows ({int(df['label'].sum()):,} spam)")


if __name__ == "__main__":
    main()
