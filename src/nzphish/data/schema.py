"""Unified dataset schema for NZ phishing/smishing.

All collectors output rows matching ``CANONICAL_COLUMNS``. Splits are stratified
by (label, channel) and pinned via SHA-256 of the sorted, serialized parquet.
"""

from __future__ import annotations

CANONICAL_COLUMNS: tuple[str, ...] = (
    "id",  # stable identifier "{source}_{n}"
    "text",  # raw message body (SMS) or subject + "\n" + body (email)
    "channel",  # "email" | "sms"
    "label",  # 0 = legitimate, 1 = phishing/smishing
    "source",  # collector name (e.g. "uci_sms", "nz_synth")
    "lang",  # ISO 639-1 best-effort, default "en"
)

LABEL_LEGIT = 0
LABEL_PHISH = 1

CHANNEL_EMAIL = "email"
CHANNEL_SMS = "sms"
