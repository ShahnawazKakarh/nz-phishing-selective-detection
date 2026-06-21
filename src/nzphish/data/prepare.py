"""Prepare unified train/val/test splits from all available collectors.

Reads per-source parquet files from ``data/processed/sources/`` (produced by the
``scripts/collect_*.py`` and ``scripts/synth_*.py`` scripts), concatenates them,
deduplicates by normalized text, stratified-splits 80/10/10 by (label, channel),
and writes:

- ``data/processed/train.parquet``
- ``data/processed/val.parquet``
- ``data/processed/test.parquet``
- ``data/processed/manifest.json``  (per-split row count + SHA-256)

Usage:
    nzphish-prepare --seed 42
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from nzphish.data.schema import CANONICAL_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES_DIR = REPO_ROOT / "data" / "processed" / "sources"
OUT_DIR = REPO_ROOT / "data" / "processed"


def _normalize_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_all_sources() -> pd.DataFrame:
    if not SOURCES_DIR.exists():
        raise FileNotFoundError(
            f"No sources at {SOURCES_DIR}. Run scripts/collect_*.py and scripts/synth_*.py first."
        )
    parts = []
    for p in sorted(SOURCES_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        missing = set(CANONICAL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} missing columns: {missing}")
        parts.append(df[list(CANONICAL_COLUMNS)])
        click.echo(f"loaded {p.name}: {len(df):,} rows")
    if not parts:
        raise RuntimeError("no parquet sources found")
    return pd.concat(parts, ignore_index=True)


@click.command()
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--val-frac", default=0.1, show_default=True, type=float)
@click.option("--test-frac", default=0.1, show_default=True, type=float)
def main(seed: int, val_frac: float, test_frac: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_all_sources()

    before = len(df)
    df["_norm"] = df["text"].map(_normalize_text)
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"]).reset_index(drop=True)
    click.echo(f"deduped: {before:,} -> {len(df):,}")

    # Stratify on (label, channel) combo
    strata = df["label"].astype(str) + "_" + df["channel"].astype(str)
    train_df, hold = train_test_split(
        df, test_size=val_frac + test_frac, stratify=strata, random_state=seed
    )
    hold_strata = hold["label"].astype(str) + "_" + hold["channel"].astype(str)
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        hold, test_size=rel_test, stratify=hold_strata, random_state=seed
    )

    manifest: dict[str, dict] = {"seed": seed, "splits": {}}
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        out = OUT_DIR / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(out, index=False)
        manifest["splits"][name] = {
            "rows": int(len(part)),
            "n_phish": int(part["label"].sum()),
            "n_email": int((part["channel"] == "email").sum()),
            "n_sms": int((part["channel"] == "sms").sum()),
            "sha256": _sha256(out),
        }
        click.echo(f"wrote {out.name}: {len(part):,} rows")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    click.echo(f"manifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
