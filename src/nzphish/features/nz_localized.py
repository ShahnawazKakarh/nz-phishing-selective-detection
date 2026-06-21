"""NZ-localized phishing cue extractors.

Each function takes a raw text (email body, SMS, URL, or concatenated subject+body)
and returns a feature dict. Designed to be cheap, deterministic, and explainable —
suitable for both standalone use and as augmentation to a TF-IDF or transformer
pipeline.

References (publicly documented NZ phishing patterns):
- CERT NZ Phishing Disruption Service quarterly reports
- Netsafe scam advisories
- Banking Ombudsman New Zealand fraud bulletins
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import tldextract

# ---------------------------------------------------------------------------
# Lexicons (publicly documented; impersonated entities only)
# ---------------------------------------------------------------------------

NZ_GOVT_ENTITIES: tuple[str, ...] = (
    "inland revenue",
    "ird",
    "te tari taake",
    "nz post",
    "new zealand post",
    "nzpost",
    "waka kotahi",
    "nzta",
    "new zealand transport agency",
    "msd",
    "work and income",
    "winz",
    "studylink",
    "acc",
    "accident compensation",
    "department of internal affairs",
    "dia",
    "ministry of health",
    "te whatu ora",
    "health new zealand",
    "immigration new zealand",
    "inz",
)

NZ_BANKS: tuple[str, ...] = (
    "anz",
    "asb",
    "westpac",
    "kiwibank",
    "bnz",
    "bank of new zealand",
    "tsb",
    "co-operative bank",
    "heartland",
    "rabobank",
)

NZ_TELCO_UTILITY: tuple[str, ...] = (
    "spark",
    "one nz",
    "vodafone nz",
    "2degrees",
    "skinny",
    "contact energy",
    "mercury",
    "genesis energy",
    "meridian",
)

TE_REO_SALUTATIONS: tuple[str, ...] = (
    "kia ora",
    "tēnā koe",
    "tena koe",
    "tēnā koutou",
    "tena koutou",
    "mōrena",
    "morena",
    "kia kaha",
    "ngā mihi",
    "nga mihi",
    "noho ora mai",
)

NZ_LURE_KEYWORDS: tuple[str, ...] = (
    "tax refund",
    "ird refund",
    "parcel held",
    "redelivery",
    "customs fee",
    "toll fee",
    "unpaid toll",
    "infringement notice",
    "fines payment",
    "covid payment",
    "winter energy",
    "rates rebate",
)

# Legitimate NZ domains commonly spoofed
LEGIT_NZ_DOMAINS: tuple[str, ...] = (
    "ird.govt.nz",
    "nzpost.co.nz",
    "nzta.govt.nz",
    "msd.govt.nz",
    "anz.co.nz",
    "asb.co.nz",
    "westpac.co.nz",
    "kiwibank.co.nz",
    "bnz.co.nz",
)

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# NZ mobile: +64 2x xxx xxxx, 02x-xxx-xxxx (separators optional)
NZ_MOBILE_RE = re.compile(
    r"(?:\+?64\s?|0)2[0-9](?:[\s\-]?\d){6,8}",
    re.IGNORECASE,
)
# NZ landline area codes: 03/04/06/07/09
NZ_LANDLINE_RE = re.compile(
    r"(?:\+?64\s?|0)[34679](?:[\s\-]?\d){7}",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
NZ_POSTCODE_RE = re.compile(r"\b[0-9]{4}\b")
NZD_RE = re.compile(r"(?:NZ\$|NZD\s?\$?|\$NZ)\s?\d+(?:[.,]\d+)?", re.IGNORECASE)

HOMOGLYPH_CHARS = set("аеіорсухӏ")  # Cyrillic look-alikes commonly used in spoofs


@dataclass
class NZFeatures:
    """Per-sample NZ phishing cue features."""

    impersonated_govt: int
    impersonated_bank: int
    impersonated_telco_utility: int
    te_reo_salutation: int
    nz_lure_keyword_count: int
    nz_mobile_count: int
    nz_landline_count: int
    nzd_mention_count: int
    nz_postcode_count: int
    url_count: int
    suspicious_govt_lookalike: int
    suspicious_bank_lookalike: int
    homoglyph_chars: int
    has_punycode: int

    def to_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def _normalize(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower()


def _count_any(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in text)


def _domain_lookalike(url: str, legit: str) -> bool:
    """True if extracted domain *resembles* a legit NZ domain but is not it."""
    try:
        ext = tldextract.extract(url)
        candidate = ".".join(p for p in (ext.domain, ext.suffix) if p)
    except Exception:
        return False
    if not candidate or candidate == legit:
        return False
    # crude similarity: legit token present in subdomain or domain but TLD differs
    legit_root = legit.split(".")[0]
    return legit_root in candidate.lower() and candidate.lower() != legit


def extract_nz_features(text: str) -> NZFeatures:
    """Extract NZ-localized phishing cues from a text.

    Args:
        text: Raw email body, SMS content, or URL string.

    Returns:
        ``NZFeatures`` with integer cue counts.
    """
    norm = _normalize(text or "")
    urls = URL_RE.findall(text or "")

    govt_lookalike = sum(
        1
        for u in urls
        for legit in LEGIT_NZ_DOMAINS
        if legit.endswith(".govt.nz") and _domain_lookalike(u, legit)
    )
    bank_lookalike = sum(
        1
        for u in urls
        for legit in LEGIT_NZ_DOMAINS
        if legit.endswith(".co.nz") and _domain_lookalike(u, legit)
    )

    homoglyphs = sum(1 for ch in text or "" if ch.lower() in HOMOGLYPH_CHARS)
    has_punycode = int("xn--" in (text or "").lower())

    return NZFeatures(
        impersonated_govt=int(_count_any(norm, NZ_GOVT_ENTITIES) > 0),
        impersonated_bank=int(_count_any(norm, NZ_BANKS) > 0),
        impersonated_telco_utility=int(_count_any(norm, NZ_TELCO_UTILITY) > 0),
        te_reo_salutation=int(_count_any(norm, TE_REO_SALUTATIONS) > 0),
        nz_lure_keyword_count=_count_any(norm, NZ_LURE_KEYWORDS),
        nz_mobile_count=len(NZ_MOBILE_RE.findall(text or "")),
        nz_landline_count=len(NZ_LANDLINE_RE.findall(text or "")),
        nzd_mention_count=len(NZD_RE.findall(text or "")),
        nz_postcode_count=len(NZ_POSTCODE_RE.findall(text or "")),
        url_count=len(urls),
        suspicious_govt_lookalike=govt_lookalike,
        suspicious_bank_lookalike=bank_lookalike,
        homoglyph_chars=homoglyphs,
        has_punycode=has_punycode,
    )


__all__ = [
    "NZFeatures",
    "extract_nz_features",
    "NZ_GOVT_ENTITIES",
    "NZ_BANKS",
    "NZ_TELCO_UTILITY",
    "TE_REO_SALUTATIONS",
    "NZ_LURE_KEYWORDS",
    "LEGIT_NZ_DOMAINS",
]
