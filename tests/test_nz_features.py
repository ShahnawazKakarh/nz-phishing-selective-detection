"""Tests for the NZ-localized feature extractor."""

from nzphish.features.nz_localized import extract_nz_features


def test_ird_refund_lure_email():
    text = (
        "Kia ora, Inland Revenue has approved your tax refund of NZ$842.50. "
        "Confirm your bank details at https://ird-refund-portal.com to receive payment. "
        "Call 021 555 1234 if you need help."
    )
    f = extract_nz_features(text)
    assert f.impersonated_govt == 1
    assert f.te_reo_salutation == 1
    assert f.nz_lure_keyword_count >= 1
    assert f.nzd_mention_count >= 1
    assert f.nz_mobile_count >= 1
    assert f.url_count >= 1


def test_nzpost_smishing():
    # Note: lexicon uses exact-substring matching by design (cheap, explainable).
    # "parcel is held" does not match "parcel held", so only "customs fee" hits here.
    text = "NZ Post: Your parcel is held due to unpaid customs fee. Pay $3.20 at https://nzpost-redeliver.xyz"
    f = extract_nz_features(text)
    assert f.impersonated_govt == 1
    assert f.nz_lure_keyword_count >= 1
    assert f.url_count >= 1


def test_multi_lure_redelivery_email():
    text = (
        "NZ Post redelivery notice: your parcel held at depot. "
        "Pay the customs fee to schedule redelivery."
    )
    f = extract_nz_features(text)
    assert f.impersonated_govt == 1
    assert f.nz_lure_keyword_count >= 2  # parcel held + customs fee + redelivery


def test_legitimate_personal_email():
    text = "Hi Sarah, just confirming dinner on Friday at 7pm. Cheers."
    f = extract_nz_features(text)
    assert f.impersonated_govt == 0
    assert f.impersonated_bank == 0
    assert f.nz_lure_keyword_count == 0
    assert f.te_reo_salutation == 0


def test_bank_phish_with_homoglyph():
    text = "АNZ Bank security alert: log in at https://аnz-secure.co to verify"
    f = extract_nz_features(text)
    assert f.homoglyph_chars >= 1
