"""Generate Gemini-paraphrased NZ lures for v0.2.1.

Reads ``data/processed/sources/nz_synth.parquet``, paraphrases each row N times
using Gemini 2.5 Flash, and writes
``data/processed/sources/nz_synth_paraphrased.parquet`` with one row per
*paraphrase* (preserving original label / channel / lang, source set to
``nz_synth_para``).

The paraphraser is disk-cached and 429-resilient: if the free-tier quota
(20 RPD / 5 RPM) is hit, the script aborts cleanly and the next run resumes
from the first uncached item.

Usage:
    python scripts/paraphrase_nz_lures.py --seed 42 --n-paraphrases 4
    # Re-run tomorrow if quota was hit; it picks up where it stopped.

Free-tier note: with N=4 paraphrases per row and 1,200 source rows, this is
1,200 API calls (one call returns N paraphrases). Free tier = 20 RPD, so this
takes ~60 days to complete on free tier; consider a paid tier or reduce the
source size with --max-rows for first runs.
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from nzphish.data.paraphrase import ParaphraseConfig, paraphrase_batch
from nzphish.data.schema import CANONICAL_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "processed" / "sources" / "nz_synth.parquet"
OUT = REPO_ROOT / "data" / "processed" / "sources" / "nz_synth_paraphrased.parquet"


@click.command()
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--n-paraphrases",
    default=4,
    show_default=True,
    type=int,
    help="Number of paraphrases to request per source row.",
)
@click.option(
    "--max-rows",
    default=-1,
    show_default=True,
    type=int,
    help="Cap source rows for quota-limited runs. -1 = all.",
)
@click.option("--model", default="gemini-2.5-flash", show_default=True)
@click.option("--temperature", default=0.9, show_default=True, type=float)
@click.option(
    "--min-interval-s",
    default=12.0,
    show_default=True,
    type=float,
    help="Minimum seconds between uncached API calls (5 RPM = 12 s).",
)
def main(
    seed: int,
    n_paraphrases: int,
    max_rows: int,
    model: str,
    temperature: float,
    min_interval_s: float,
) -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} — run scripts/synth_nz_lures.py first")

    src = pd.read_parquet(SRC)
    click.echo(f"loaded {SRC.name}: {len(src):,} rows")
    if max_rows > 0 and max_rows < len(src):
        src = src.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        click.echo(f"capped to {len(src):,} rows for this run")

    cfg = ParaphraseConfig(
        model=model,
        n_paraphrases=n_paraphrases,
        temperature=temperature,
    )
    paraphrases_per_row = paraphrase_batch(
        src["text"].astype(str).tolist(),
        cfg=cfg,
        seed=seed,
        min_interval_s=min_interval_s,
    )

    rows: list[dict] = []
    n_skipped = 0
    for orig_row, paraphrases in zip(src.to_dict("records"), paraphrases_per_row, strict=True):
        if not paraphrases:
            n_skipped += 1
            continue
        for j, p in enumerate(paraphrases):
            rows.append(
                {
                    "id": f"{orig_row['id']}__para_{j}",
                    "text": p,
                    "channel": orig_row["channel"],
                    "label": orig_row["label"],
                    "source": "nz_synth_para",
                    "lang": orig_row["lang"],
                }
            )

    if not rows:
        click.echo("no paraphrases generated (quota exhausted on first call?). nothing written.")
        return

    df = pd.DataFrame(rows)[list(CANONICAL_COLUMNS)]
    df.to_parquet(OUT, index=False)
    click.echo(
        f"wrote {OUT}: {len(df):,} paraphrased rows "
        f"(from {len(src) - n_skipped}/{len(src)} source rows; "
        f"{n_skipped} skipped due to quota or empty response)"
    )


if __name__ == "__main__":
    main()
