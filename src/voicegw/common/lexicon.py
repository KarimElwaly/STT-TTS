"""Pronunciation and dialect respelling lexicon.

Normalizes unvowelized acronyms, foreign terms, numerals, and dialectal
words before synthesis to ensure natural pronunciation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def normalize_lexicon(raw: Optional[dict]) -> dict[str, str]:
    """Return a clean {key: respelling} dictionary with stripped strings."""
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, str] = {}
    for k, v in raw.items():
        if k is None or v is None:
            continue
        key = str(k).strip()
        val = str(v).strip()
        if key:
            clean[key] = val
    return clean


def compile_lexicon(lexicon: dict[str, str]) -> tuple[Optional[re.Pattern], dict[str, str]]:
    """Build a ReDoS-safe alternation regex and lookup table.

    Keys are matched longest-first so compound terms take precedence
    over individual words. Matching is performed in a single pass.
    """
    clean = normalize_lexicon(lexicon)
    if not clean:
        return None, {}

    # Sort longest key first
    sorted_keys = sorted(clean.keys(), key=len, reverse=True)
    lookup: dict[str, str] = {k.lower(): clean[k] for k in sorted_keys}

    # Build alternation
    patterns = []
    for k in sorted_keys:
        escaped = re.escape(k)
        # Apply word boundaries if key starts/ends with alphanumeric characters
        prefix = r"\b" if k[:1].isalnum() else ""
        suffix = r"\b" if k[-1:].isalnum() else ""
        patterns.append(f"{prefix}{escaped}{suffix}")

    combined = re.compile("|".join(patterns), re.IGNORECASE)
    return combined, lookup


def apply_lexicon(text: str, lexicon: Optional[dict[str, str]]) -> str:
    """Apply pronunciation replacements to text in a single ReDoS-safe pass."""
    if not text or not lexicon:
        return text

    pattern, lookup = compile_lexicon(lexicon)
    if pattern is None:
        return text

    def _replace(match: re.Match) -> str:
        word = match.group(0).lower()
        return lookup.get(word, match.group(0))

    return pattern.sub(_replace, text)


class LexiconManager:
    """Manages project-wide pronunciation lexicon files."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else Path("voices/lexicon.json")
        self._lexicon: dict[str, str] = {}
        self._pattern: Optional[re.Pattern] = None
        self._lookup: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._lexicon = {}
            self._pattern, self._lookup = None, {}
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._lexicon = normalize_lexicon(raw)
            self._pattern, self._lookup = compile_lexicon(self._lexicon)
            log.info("Loaded %d pronunciation lexicon rules from %s", len(self._lexicon), self.path)
        except Exception as exc:
            log.warning("Could not read lexicon at %s: %s", self.path, exc)
            self._lexicon = {}
            self._pattern, self._lookup = None, {}

    def apply(self, text: str) -> str:
        if not text or self._pattern is None:
            return text

        def _replace(match: re.Match) -> str:
            word = match.group(0).lower()
            return self._lookup.get(word, match.group(0))

        return self._pattern.sub(_replace, text)


_default_lexicon_mgr: LexiconManager | None = None


def get_lexicon_manager() -> LexiconManager:
    global _default_lexicon_mgr
    if _default_lexicon_mgr is None:
        _default_lexicon_mgr = LexiconManager()
    return _default_lexicon_mgr
