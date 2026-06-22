"""TF-IDF + Logistic Regression baseline with optional NZ-localized features.

The NZ feature columns from ``nzphish.features.nz_localized`` are concatenated
with the TF-IDF vector via ``scipy.sparse.hstack``. This gives an interpretable
linear baseline where the contribution of NZ cues is directly inspectable from
the model's coefficients.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nzphish.features.nz_localized import extract_nz_features


@dataclass
class TfidfLRConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.95
    max_features: int | None = 50_000
    sublinear_tf: bool = True
    use_nz_features: bool = True
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    nz_feature_names: list[str] = field(default_factory=lambda: [
        "impersonated_govt", "impersonated_bank", "impersonated_telco_utility",
        "te_reo_salutation", "nz_lure_keyword_count", "nz_mobile_count",
        "nz_landline_count", "nzd_mention_count", "nz_postcode_count",
        "url_count", "suspicious_govt_lookalike", "suspicious_bank_lookalike",
        "homoglyph_chars", "has_punycode",
    ])


def _nz_feature_matrix(texts: list[str], names: list[str]) -> sparse.csr_matrix:
    rows = [extract_nz_features(t).to_dict() for t in texts]
    arr = np.array([[r[k] for k in names] for r in rows], dtype=np.float32)
    return sparse.csr_matrix(arr)


class TfidfLR:
    """TF-IDF + optional NZ cue features + Logistic Regression."""

    def __init__(self, cfg: TfidfLRConfig):
        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(
            ngram_range=(cfg.ngram_min, cfg.ngram_max),
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            max_features=cfg.max_features,
            sublinear_tf=cfg.sublinear_tf,
            lowercase=True,
            strip_accents="unicode",
        )
        self.clf = LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
        )

    def _featurize(self, texts: list[str], fit: bool) -> sparse.csr_matrix:
        tfidf = self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        if not self.cfg.use_nz_features:
            return tfidf
        nz = _nz_feature_matrix(texts, self.cfg.nz_feature_names)
        return sparse.hstack([tfidf, nz]).tocsr()

    def fit(self, texts: list[str], labels: np.ndarray) -> "TfidfLR":
        X = self._featurize(texts, fit=True)
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._featurize(texts, fit=False)
        return self.clf.predict_proba(X)

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] >= threshold).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"cfg": self.cfg, "vectorizer": self.vectorizer, "clf": self.clf}, f)

    @classmethod
    def load(cls, path: Path) -> "TfidfLR":
        with path.open("rb") as f:
            obj = pickle.load(f)
        m = cls(obj["cfg"])
        m.vectorizer = obj["vectorizer"]
        m.clf = obj["clf"]
        return m


def fit_from_df(train_df: pd.DataFrame, cfg: TfidfLRConfig) -> TfidfLR:
    model = TfidfLR(cfg)
    model.fit(train_df["text"].astype(str).tolist(), train_df["label"].to_numpy())
    return model
