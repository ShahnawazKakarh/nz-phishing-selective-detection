"""Smoke test for the TF-IDF + LR baseline on tiny in-memory data."""

import numpy as np
import pandas as pd

from nzphish.models.tfidf_lr import TfidfLR, TfidfLRConfig


def test_tfidf_lr_fits_and_predicts():
    train = pd.DataFrame(
        {
            "text": [
                "Kia ora, IRD tax refund of NZ$842 approved, click https://ird-refund.com",
                "NZ Post parcel held customs fee https://nzpost-fake.xyz",
                "ANZ login alert verify at https://anz-secure.co",
                "Hi Sarah dinner Friday 7pm cheers",
                "Reminder appointment tomorrow at 3pm",
                "Hey running 10 min late see you soon",
            ],
            "label": [1, 1, 1, 0, 0, 0],
        }
    )
    model = TfidfLR(TfidfLRConfig(min_df=1, max_features=200))
    model.fit(train["text"].tolist(), train["label"].to_numpy())
    probs = model.predict_proba(train["text"].tolist())[:, 1]
    assert probs.shape == (6,)
    # On the training set, phish samples should score higher than ham on average
    assert probs[:3].mean() > probs[3:].mean()


def test_tfidf_lr_no_nz_features_branch():
    train = pd.DataFrame(
        {
            "text": ["spam phish click", "hello friend dinner"],
            "label": [1, 0],
        }
    )
    model = TfidfLR(TfidfLRConfig(min_df=1, max_features=50, use_nz_features=False))
    model.fit(train["text"].tolist(), np.array([1, 0]))
    p = model.predict_proba(["click phish now"])
    assert p.shape == (1, 2)
