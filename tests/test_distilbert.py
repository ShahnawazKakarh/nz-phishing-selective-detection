"""Smoke test for the DistilBERT wrapper. Marked slow; skipped in CI."""

import numpy as np
import pytest

from nzphish.models.distilbert import DistilBertClassifier, DistilBertConfig


@pytest.mark.slow
def test_distilbert_fits_and_predicts_tiny():
    cfg = DistilBertConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        max_length=64,
        max_train_samples=8,
    )
    model = DistilBertClassifier(cfg)
    texts = [
        "Kia ora, IRD refund of NZ$842 click https://ird-refund.com to confirm",
        "NZ Post parcel held, pay customs fee https://nzpost-fake.xyz",
        "ANZ security alert verify your login https://anz-secure.co",
        "Westpac suspicious login, confirm at https://westpac-nz-fake.net",
        "Hi Sarah, dinner Friday 7pm cheers",
        "Reminder: appointment tomorrow at 3pm",
        "Hey running 10 min late see you soon",
        "Kia ora team, ngā mihi for the hui notes",
    ]
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    model.fit(texts, labels)
    probs = model.predict_proba(texts)
    assert probs.shape == (8, 2)
    assert np.all((probs >= 0) & (probs <= 1))
