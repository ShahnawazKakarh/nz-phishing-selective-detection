"""Gemini 2.5 Flash paraphraser with disk cache and quota-resilient resume.

Designed for the Gemini API free tier (20 RPD, 5 RPM). When a 429 quota error
hits, the function aborts cleanly without losing prior work — every successful
paraphrase is cached to disk by ``sha256(text + prompt + seed + index)`` and a
subsequent run resumes from the first uncached row.

This mirrors the pattern used in the author's `hate-speech-detection v0.2.1`
LLM-as-classifier pipeline (see ORCID 0009-0007-4055-6563).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "llm_cache" / "paraphrase"

DEFAULT_MODEL = "gemini-2.5-flash"

PARAPHRASE_PROMPT = (
    "You are paraphrasing a New Zealand phishing or legitimate message for an "
    "academic NLP research benchmark. The paraphrase must:\n"
    "1. Preserve the exact intent (phishing vs legitimate) and the target NZ "
    "entity if one is named (IRD, NZ Post, Waka Kotahi/NZTA, ANZ, ASB, "
    "Westpac, Kiwibank, BNZ).\n"
    "2. Change wording, sentence structure, and surface phrasing significantly "
    "so a bag-of-words classifier cannot trivially memorize the template.\n"
    "3. Keep the channel-appropriate length (SMS short, email longer).\n"
    "4. Keep any URLs verbatim if present.\n"
    "5. Stay in English (may include te reo Māori greetings only if the "
    "original used them).\n\n"
    "Return ONLY a JSON object with key 'paraphrases', whose value is a list "
    "of {n} paraphrased strings. No extra commentary."
)


@dataclass
class ParaphraseConfig:
    model: str = DEFAULT_MODEL
    n_paraphrases: int = 4
    temperature: float = 0.9
    max_output_tokens: int = 2048
    thinking_budget: int = 0
    request_timeout_s: int = 60


def _key(text: str, cfg: ParaphraseConfig, seed: int) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(cfg.model.encode())
    h.update(str(cfg.n_paraphrases).encode())
    h.update(str(cfg.temperature).encode())
    h.update(str(seed).encode())
    return h.hexdigest()


def _load_cached(key: str) -> list[str] | None:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["paraphrases"]
    except Exception:
        return None


def _save_cached(key: str, paraphrases: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps({"paraphrases": paraphrases}))


def _client() -> genai.Client:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing — add it to .env")
    return genai.Client(api_key=key)


class QuotaExhausted(RuntimeError):
    """Raised when the Gemini API returns 429 so callers can abort gracefully."""


def paraphrase_one(client: genai.Client, text: str, cfg: ParaphraseConfig, seed: int) -> list[str]:
    """Paraphrase a single text into ``cfg.n_paraphrases`` variants.

    Disk-cached. Re-uses prior result if the same (text, cfg, seed) was seen.
    """
    key = _key(text, cfg, seed)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    prompt = (
        PARAPHRASE_PROMPT.format(n=cfg.n_paraphrases)
        + "\n\nORIGINAL:\n"
        + text
        + '\n\nRespond with valid JSON only: {"paraphrases": ["...", "..."]}'
    )
    try:
        resp = client.models.generate_content(
            model=cfg.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
                response_mime_type="application/json",
                seed=seed,
            ),
        )
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            raise QuotaExhausted(str(e)) from e
        raise

    raw = (resp.text or "").strip()
    try:
        obj = json.loads(raw)
        paraphrases = list(obj.get("paraphrases", []))
    except json.JSONDecodeError:
        paraphrases = []

    # Filter degenerate outputs
    paraphrases = [p.strip() for p in paraphrases if isinstance(p, str) and p.strip()]
    paraphrases = [p for p in paraphrases if p.lower() != text.lower()]

    if paraphrases:
        _save_cached(key, paraphrases)
    return paraphrases


def paraphrase_batch(
    texts: list[str],
    cfg: ParaphraseConfig,
    seed: int = 42,
    min_interval_s: float = 12.0,
    progress_every: int = 25,
) -> list[list[str]]:
    """Paraphrase a batch of texts. Cache-aware, 429-resilient.

    Args:
        min_interval_s: Minimum wall-clock seconds between *uncached* API calls.
            Default 12 s targets ~5 RPM (the free-tier hard cap).
        progress_every: Print progress every N items.

    Returns:
        List of paraphrase lists, one per input text. Empty lists indicate the
        item is still uncached AND a quota error was hit (rerun later to resume).
    """
    client = _client()
    out: list[list[str]] = []
    last_call_t = 0.0
    new_calls = 0
    for i, text in enumerate(texts):
        cached = _load_cached(_key(text, cfg, seed))
        if cached is not None:
            out.append(cached)
            continue
        # Rate-limit uncached calls
        delta = time.time() - last_call_t
        if delta < min_interval_s:
            time.sleep(min_interval_s - delta)
        print(f"[paraphrase] calling Gemini for item {i + 1}/{len(texts)}...", flush=True)
        try:
            paraphrases = paraphrase_one(client, text, cfg, seed)
        except QuotaExhausted as e:
            print(f"[paraphrase] quota exhausted at item {i} ({new_calls} new calls): {e}")
            # Fill remaining with empty placeholders so caller knows what's missing
            out.append([])
            out.extend([[] for _ in range(len(texts) - i - 1)])
            return out
        last_call_t = time.time()
        new_calls += 1
        out.append(paraphrases)
        if (i + 1) % progress_every == 0:
            print(f"[paraphrase] {i + 1}/{len(texts)} done ({new_calls} new API calls)")
    print(f"[paraphrase] complete: {len(texts)} items, {new_calls} new API calls")
    return out
