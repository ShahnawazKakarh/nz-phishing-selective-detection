# nz-phishing-selective-detection

**NZ-localized phishing and smishing detection with class-conditional selective abstention.**

This repository accompanies an independent research study targeting the New Zealand
phishing and smishing threat landscape. It combines an NZ-localized feature set
(Inland Revenue, NZ Post, NZTA, ANZ/ASB/Westpac/Kiwibank/BNZ lookalikes, `.govt.nz`
domain spoofing, te reo Māori salutations, NZ phone-number formats) with three
classifier families and a selective-prediction layer that abstains on borderline
samples and routes them to human review.

> **Status:** v0.1.0 scaffold. Data collection and baselines in progress.
> **Author:** Muhammad Shahnawaz Khan (ORCID
> [0009-0007-4055-6563](https://orcid.org/0009-0007-4055-6563))

---

## Research contributions (planned for v1.0.0)

1. **NZ-localized phishing/smishing corpus.** Curated from CERT NZ public advisories,
   Netsafe reports, PhishTank/Nazario filtered to NZ-relevant lures, augmented with
   NZ-specific synthetic lures (te reo Māori, NZ entity branding, NZ phone formats).
2. **Three-classifier benchmark.** TF-IDF + Logistic Regression, fine-tuned
   DistilBERT, and Gemini 2.5 Flash LLM-as-classifier (zero-shot and few-shot).
3. **Class-conditional selective abstention.** A deferral layer (extending LW-CCSD
   and OACSP from prior work) that learns class-specific confidence thresholds and
   abstains on inputs likely to be misclassified, surfacing them for analyst review.
4. **Robustness audit.** Character-level perturbation and homoglyph attacks (common
   in real NZ phishing) measured against AUC and selective F1.
5. **Bias and coverage analysis.** Disaggregated performance across email vs SMS,
   English vs te reo Māori, and per-impersonated-entity (IRD, NZ Post, banks).

## Why NZ-specific?

Global phishing detectors miss NZ-specific lures: IRD tax refunds, NZ Post parcel
redelivery scams, Waka Kotahi (NZTA) toll/infringement scams, and bank-specific
language patterns. The novel contribution here is not a new architecture but a
defensible, reproducible NZ benchmark with calibrated abstention — directly usable
by CERT NZ, Netsafe, and the NZ banking sector.

## Repository layout

```
nz-phishing-selective-detection/
├── src/nzphish/
│   ├── data/         # collection, cleaning, splits
│   ├── features/     # NZ-localized feature extractors
│   ├── models/       # TF-IDF+LR, DistilBERT, Gemini LLM wrapper
│   ├── deferral/     # class-conditional selective prediction
│   ├── eval/         # metrics, selective metrics, robustness
│   └── utils/
├── configs/          # YAML configs per model
├── scripts/          # one-shot data + train + eval entry points
├── data/             # raw/, processed/, llm_cache/ (all gitignored)
├── docs/baselines/   # pinned baseline results per version
├── paper/            # IEEE-style LaTeX source
├── results/          # metrics tables, plots
└── tests/
```

## Quickstart

```bash
pyenv install 3.11.1   # if missing
pyenv local 3.11.1
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add GEMINI_API_KEY
```

## Data sources

| Source                       | Type            | Notes                                         |
|------------------------------|-----------------|-----------------------------------------------|
| CERT NZ Phishing Disruption  | URLs / lures    | Public advisories at cert.govt.nz             |
| Netsafe scam reports         | Email / SMS     | Public summaries                              |
| Nazario phishing corpus      | Email           | Global, filtered for NZ relevance             |
| PhishTank verified           | URLs            | Filtered for NZ-targeted brands               |
| SMS Phishing Dataset (UCI)   | SMS             | Global SMS baseline                           |
| Enron + SpamAssassin Ham     | Email           | Legitimate class                              |
| NZ synthetic augmentation    | Email / SMS     | Generated per documented templates            |

All collection scripts under `scripts/collect_*.py`. Raw data is **never** committed;
processed splits are versioned with their checksum in `docs/baselines/v0.x.x/`.

## Reproducibility

- Python 3.11 pinned via `.python-version`.
- All training entry points accept `--seed` and log to `results/runs/`.
- Baselines pinned in `docs/baselines/v{version}/` with config + metrics + checksum.
- Zenodo DOI on first tagged release.

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff). On release, this repository will be archived to
Zenodo with a permanent DOI.

## Disclaimer

This repository is independent research. It is **not** affiliated with CERT NZ,
Netsafe, any New Zealand government agency, or any New Zealand bank. References to
NZ entities are limited to publicly documented phishing patterns and are used solely
for defensive research.
