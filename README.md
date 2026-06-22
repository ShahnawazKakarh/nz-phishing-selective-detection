# nz-phishing-selective-detection

[![CI](https://github.com/ShahnawazKakarh/nz-phishing-selective-detection/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ShahnawazKakarh/nz-phishing-selective-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)

**NZ-localized phishing and smishing detection with class-conditional selective abstention.**

Independent research repository targeting the New Zealand phishing and smishing threat landscape. The project couples a hand-engineered NZ-localized feature extractor (Inland Revenue, NZ Post, Waka Kotahi, ANZ/ASB/Westpac/Kiwibank/BNZ lookalikes, `.govt.nz` homoglyphs, te reo Māori salutations, NZ phone-number formats) with a multi-family classifier benchmark and a planned selective-prediction layer for high-stakes deployment.

> **Status:** v0.2.0 — two classifier families (TF-IDF + LR, DistilBERT) benchmarked end-to-end with stratified evaluation. v0.2.1 adds Gemini-paraphrased lures and LLM-as-classifier. Selective deferral lands in v0.3.0.
> **Author:** Muhammad Shahnawaz Khan ([ORCID 0009-0007-4055-6563](https://orcid.org/0009-0007-4055-6563))

---

## Why this project

Global phishing detectors miss NZ-specific lures: IRD tax refunds, NZ Post parcel redelivery scams, Waka Kotahi (NZTA) toll/infringement notices, and bank-specific impersonation patterns. This repository builds a defensible, reproducible NZ benchmark with calibrated abstention — directly usable by CERT NZ, Netsafe, and the NZ banking sector for triage and human-in-the-loop review.

## Research contributions

1. **NZ-localized feature extractor** (`src/nzphish/features/nz_localized.py`) — 14 explainable cues covering impersonated NZ entities, te reo salutations, NZ lure keywords, NZ phone-number patterns, `.govt.nz` / `.co.nz` lookalike detection, and homoglyph / punycode flags.
2. **Multi-family classifier benchmark** — TF-IDF + Logistic Regression (with and without NZ features as ablation), DistilBERT fine-tune, and (v0.2.1) Gemini 2.5 Flash as LLM-as-classifier under zero-shot and few-shot prompts.
3. **Reproducible NZ corpus** — UCI SMS Spam (5,574 SMS) as the global baseline plus a deterministic NZ synthetic lure generator covering four documented NZ lure families. v0.2.1 extends this with LLM-paraphrased variants and adversarial NZ negatives.
4. **Stratified evaluation** — per-source and per-channel metrics so the NZ-features contribution can be measured against the global-spam subset rather than hidden by aggregate averages.
5. **Class-conditional selective abstention** (planned v0.3.0) — extending LW-CCSD (rPPG AF screening) and OACSP (retinal DR grading) prior work by the same author to binary phishing classification under operator-specified coverage constraints.

## Results

### Aggregate test set (n=583, threshold 0.5)

| Model                                | F1-phish | Precision | Recall | ROC-AUC | Accuracy | Fit time |
|--------------------------------------|----------|-----------|--------|---------|----------|----------|
| TF-IDF + LR (with NZ features)       | 0.9730   | 1.0000    | 0.9474 | 0.9993  | 0.9897   | 0.3 s    |
| TF-IDF + LR (no NZ, ablation)        | 0.9778   | 0.9910    | 0.9649 | 0.9987  | 0.9914   | 0.1 s    |
| **DistilBERT fine-tune** (3 epochs)  | **0.9782** | 0.9739 | **0.9825** | **0.9997** | **0.9914** | 329 s |

### Stratified — `source=uci_sms` (n=517, the only non-saturated subgroup)

| Model              | F1-phish | ROC-AUC |
|--------------------|----------|---------|
| TF-IDF + NZ        | 0.9500   | 0.9987  |
| TF-IDF no NZ       | 0.9593   | 0.9976  |
| **DistilBERT**     | **0.9606** | **0.9995** |

All three models reach 1.000 F1 on the NZ synthetic subset (n=66), reflecting a ceiling effect on the v0.1.0–v0.2.0 synthetic corpus rather than ideal real-world performance. v0.2.1 introduces LLM-paraphrased lures so the NZ-features contribution becomes measurable. Full per-subgroup tables are pinned in [`docs/baselines/v0.2.0.md`](docs/baselines/v0.2.0.md).

## Quickstart

```bash
git clone https://github.com/ShahnawazKakarh/nz-phishing-selective-detection
cd nz-phishing-selective-detection
pyenv local 3.11.1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add GEMINI_API_KEY when running v0.2.1+ steps
```

End-to-end reproduction of the v0.2.0 baseline:

```bash
# 1. Data
python scripts/collect_uci_sms.py
python scripts/synth_nz_lures.py --seed 42 --n-per-class-channel 300
nzphish-prepare --seed 42

# 2. Models
nzphish-train --config configs/tfidf_lr.yaml        --seed 42
nzphish-train --config configs/tfidf_lr_no_nz.yaml  --seed 42
nzphish-train --config configs/distilbert.yaml      --seed 42

# 3. Stratified eval (substitute your run directory name)
nzphish-eval-stratified --run-dir results/runs/<run_dir> --split test
```

## Repository layout

```
nz-phishing-selective-detection/
├── src/nzphish/
│   ├── data/         # canonical schema, dataset prepare CLI
│   ├── features/     # NZ-localized cue extractors
│   ├── models/       # TF-IDF + LR, DistilBERT (Gemini in v0.2.1)
│   ├── deferral/     # class-conditional selective prediction (v0.3.0)
│   ├── eval/         # classification + selective metrics, stratified runner
│   └── utils/
├── configs/          # YAML config per model
├── scripts/          # one-shot data collection and synthesis
├── data/             # raw/, processed/, llm_cache/ (gitignored)
├── docs/baselines/   # pinned per-version baseline reports
├── paper/            # IEEE-style LaTeX source (v0.3.0+)
├── results/runs/     # per-run metrics, probs, fitted models (gitignored)
└── tests/
```

## Data sources

| Source                    | Type     | Volume          | License / access            |
|---------------------------|----------|------------------|------------------------------|
| UCI SMS Spam Collection   | SMS      | 5,574 messages   | UCI ML Repository, public    |
| NZ synthetic lures (v0.1) | E + SMS  | 1,200 messages   | Generated, MIT, deterministic|
| NZ synthetic (v0.2.1)     | E + SMS  | LLM-paraphrased  | Generated, MIT               |
| Nazario phishing emails   | Email    | TBD (v0.2.1)     | Public, NZ-relevance filtered|
| PhishTank verified URLs   | URL      | TBD (v0.2.1)     | Public, NZ-brand filtered    |

Raw corpora are never committed. Processed splits are versioned by SHA-256 in `data/processed/manifest.json` and pinned per version in `docs/baselines/`.

## Reproducibility

- Python 3.11 pinned via `.python-version`; CI tests on 3.11 and 3.12.
- Every training entry point accepts `--seed` and writes per-sample probabilities for downstream calibration.
- Pinned baselines per version in `docs/baselines/v{version}.md` with config, metrics, and dataset checksums.
- `ruff==0.15.18` pinned exact-version for deterministic linting across local and CI.
- Zenodo DOI on first tagged release (v0.3.0).

## Roadmap

- **v0.2.1** — Gemini-paraphrased NZ lures; adversarial NZ negatives; Gemini 2.5 Flash LLM-as-classifier (zero-shot + 4-shot); Nazario + PhishTank NZ-filtered subsets.
- **v0.3.0** — Class-conditional selective abstention layer (LW-CCSD / OACSP extension); coverage / selective-F1 curves at multiple operating points; robustness audit (character-level perturbation, homoglyph attacks).
- **v1.0.0** — Paper submission (IEEE Access or ACM DTRAP); Zenodo + arXiv archival under "Muhammad Shahnawaz Khan".

## Contributing

This is an independent research repository and is not currently accepting external contributions. Bug reports and reproducibility issues via GitHub Issues are welcome.

If a NZ cybersecurity practitioner or researcher wishes to share documented phishing samples for inclusion (with appropriate consent and provenance), please open an issue rather than emailing — public discussion supports the research record.

## Citation

If you use this repository in academic work, please cite it via [`CITATION.cff`](CITATION.cff):

```bibtex
@software{khan_nz_phishing_selective_2026,
  author       = {Khan, Muhammad Shahnawaz},
  title        = {nz-phishing-selective-detection: NZ-localized phishing and
                  smishing detection with class-conditional selective abstention},
  year         = 2026,
  version      = {0.2.0},
  url          = {https://github.com/ShahnawazKakarh/nz-phishing-selective-detection},
  orcid        = {0009-0007-4055-6563}
}
```

On first tagged release the repository will be archived to Zenodo with a permanent DOI; the citation above will be updated accordingly.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Independent research. **Not affiliated** with CERT NZ, Netsafe, any New Zealand government agency, or any New Zealand bank. References to NZ entities are limited to publicly documented phishing patterns and are used solely for defensive research. No proprietary data is included; all corpora are public or synthetically generated.
