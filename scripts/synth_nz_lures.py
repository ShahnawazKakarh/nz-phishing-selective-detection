"""Synthesize NZ-localized phishing/smishing lures from documented templates.

Templates are derived from publicly documented NZ phishing patterns published by
CERT NZ, Netsafe, and the Banking Ombudsman New Zealand. They cover the four
dominant NZ lure families:

  1. Inland Revenue (IRD) tax refund / overpayment
  2. NZ Post parcel redelivery / customs fee
  3. Waka Kotahi (NZTA) toll / infringement
  4. NZ bank account verification / suspicious-login (ANZ/ASB/Westpac/Kiwibank/BNZ)

We also generate legitimate paired messages (NZ-flavored) so the model is not
forced to learn "any mention of IRD = phish". The deterministic ``--seed`` flag
makes the output reproducible.

Output: ``data/processed/sources/nz_synth.parquet``
"""

from __future__ import annotations

import random
from pathlib import Path

import click
import pandas as pd

from nzphish.data.schema import (
    CANONICAL_COLUMNS,
    CHANNEL_EMAIL,
    CHANNEL_SMS,
    LABEL_LEGIT,
    LABEL_PHISH,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "processed" / "sources" / "nz_synth.parquet"

# ---------------------------------------------------------------------------
# Slot fillers
# ---------------------------------------------------------------------------

NZ_FIRST_NAMES = [
    "Aroha",
    "Hemi",
    "Mere",
    "Tane",
    "Wiremu",
    "Anahera",
    "Tama",
    "Moana",
    "James",
    "Sarah",
    "Emma",
    "Liam",
    "Olivia",
    "Charlotte",
    "William",
    "Jack",
]
NZD_AMOUNTS = ["$3.20", "$8.50", "$12.40", "$48.90", "$129.00", "$284.50", "$842.30", "$1,240.00"]
NZ_MOBILES = ["021 555 1234", "022 410 7788", "027 333 9021", "029 818 4456"]
PHISH_DOMAINS_GOVT = [
    "ird-refund.com",
    "ird-portal.co",
    "ird-nz-tax.net",
    "nzpost-redeliver.xyz",
    "nzpost-track.help",
    "nzta-tolls.co",
    "waka-kotahi-fines.xyz",
]
PHISH_DOMAINS_BANK = [
    "anz-secure-login.co",
    "asb-verify-account.com",
    "westpac-nz-secure.net",
    "kiwibank-verify.xyz",
    "bnz-login-secure.co",
]
LEGIT_DOMAINS = [
    "ird.govt.nz",
    "nzpost.co.nz",
    "nzta.govt.nz",
    "anz.co.nz",
    "asb.co.nz",
    "westpac.co.nz",
    "kiwibank.co.nz",
    "bnz.co.nz",
]

# ---------------------------------------------------------------------------
# Phishing templates (label=1)
# ---------------------------------------------------------------------------

PHISH_EMAIL_TEMPLATES = [
    # IRD tax refund
    "Kia ora {name},\n\nInland Revenue has approved your tax refund of NZ${amt}. "
    "To receive payment, confirm your bank details at https://{dom}/refund within 24 hours. "
    "Failure to confirm will forfeit the refund.\n\nIRD",
    "Tēnā koe {name}, our records show an IRD overpayment of NZ${amt} is owed to you. "
    "Click https://{dom}/claim to claim before this Friday.",
    # NZ Post parcel
    "NZ Post: parcel held at depot pending an unpaid customs fee of NZD {amt}. "
    "Pay via https://{dom}/redeliver to schedule redelivery within 48 hours.",
    "Hi {name}, your NZ Post redelivery requires a small customs fee ({amt}). "
    "Confirm payment at https://{dom}/track to release your parcel.",
    # NZTA toll / infringement
    "Waka Kotahi: an unpaid toll fee of NZ${amt} is associated with your vehicle. "
    "Settle the infringement notice at https://{dom}/pay to avoid additional fines.",
    # Bank verification
    "{name}, we detected a suspicious login on your {bank} account. "
    "Verify your identity at https://{dom}/secure to prevent suspension.",
    "{bank} security alert: your account has been temporarily restricted. "
    "Log in at https://{dom}/login to restore access.",
]

PHISH_SMS_TEMPLATES = [
    "IRD: refund of {amt} approved. Confirm bank details: https://{dom}/r",
    "NZ Post: parcel held, unpaid customs fee {amt}. Pay https://{dom}/p",
    "NZTA toll fee {amt} unpaid. Avoid fines: https://{dom}/t",
    "{bank}: suspicious login detected. Verify now https://{dom}/v",
    "Waka Kotahi infringement {amt} due. Pay https://{dom}/i",
    "Kia ora, your NZ Post redelivery requires {amt}. https://{dom}/d",
]

# ---------------------------------------------------------------------------
# Legitimate templates (label=0) — NZ-flavored ham
# ---------------------------------------------------------------------------

LEGIT_EMAIL_TEMPLATES = [
    "Hi {name}, just confirming the team lunch on Friday at the cafe on Lambton Quay. "
    "Let me know if you can make it. Cheers.",
    "Tēnā koe {name}, ngā mihi for joining the hui yesterday. The minutes are attached.",
    "Hi {name}, your power bill for this month is available in your {dom} account. "
    "Log in at https://{dom} when you have a chance.",
    "Kia ora {name}, the rates notice for your property has been issued. "
    "Pay via internet banking using the reference on the notice.",
    "Hi {name}, your appointment at the clinic is confirmed for next Tuesday at 10am.",
]

LEGIT_SMS_TEMPLATES = [
    "Hey {name}, running 10 min late. See you soon.",
    "{bank}: payment of {amt} to Countdown received. Available bal updated.",
    "Reminder: your appointment is tomorrow at 3pm. Reply CANCEL to cancel.",
    "Kia ora {name}, your prescription is ready for collection.",
    "Your Uber driver will arrive in 2 minutes in a white Toyota Corolla.",
]

NZ_BANKS_DISPLAY = ["ANZ", "ASB", "Westpac", "Kiwibank", "BNZ"]


def _fill(template: str, rng: random.Random, phish: bool) -> str:
    return template.format(
        name=rng.choice(NZ_FIRST_NAMES),
        amt=rng.choice(NZD_AMOUNTS).lstrip("$"),
        dom=rng.choice(PHISH_DOMAINS_GOVT + PHISH_DOMAINS_BANK)
        if phish
        else rng.choice(LEGIT_DOMAINS),
        bank=rng.choice(NZ_BANKS_DISPLAY),
    )


@click.command()
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--n-per-class-channel",
    default=300,
    show_default=True,
    type=int,
    help="Number of samples per (label, channel) cell. Total = 4 * N.",
)
def main(seed: int, n_per_class_channel: int) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows: list[dict] = []
    cells = [
        (CHANNEL_EMAIL, LABEL_PHISH, PHISH_EMAIL_TEMPLATES, True),
        (CHANNEL_SMS, LABEL_PHISH, PHISH_SMS_TEMPLATES, True),
        (CHANNEL_EMAIL, LABEL_LEGIT, LEGIT_EMAIL_TEMPLATES, False),
        (CHANNEL_SMS, LABEL_LEGIT, LEGIT_SMS_TEMPLATES, False),
    ]
    for channel, label, templates, is_phish in cells:
        for i in range(n_per_class_channel):
            tpl = rng.choice(templates)
            text = _fill(tpl, rng, phish=is_phish)
            rows.append(
                {
                    "id": f"nz_synth_{channel}_{label}_{i}",
                    "text": text,
                    "channel": channel,
                    "label": label,
                    "source": "nz_synth",
                    "lang": "en",
                }
            )

    df = pd.DataFrame(rows)[list(CANONICAL_COLUMNS)]
    df.to_parquet(OUT, index=False)
    click.echo(
        f"wrote {OUT}: {len(df):,} rows "
        f"({int(df['label'].sum()):,} phish, "
        f"{(df['channel'] == CHANNEL_EMAIL).sum():,} email, "
        f"{(df['channel'] == CHANNEL_SMS).sum():,} sms)"
    )


if __name__ == "__main__":
    main()
