"""Voice registry.

Callers say ``voice="omar"`` instead of shipping a reference clip with every
request. The manifest maps voice ids to either a cloning reference
(``ref_audio`` + ``ref_text``) or a voice-design ``description``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..common.config import get_settings
from ..common.protocols import Voice

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 15.0

#: OmniVoice voice-design attributes. `instruct` is NOT free-form prose: it
#: must be a comma-separated list drawn from this vocabulary, or generation
#: raises ValueError at synthesis time. Validating at load time turns a
#: runtime failure into a startup warning.
INSTRUCT_VOCAB = {
    "american accent",
    "australian accent",
    "british accent",
    "canadian accent",
    "child",
    "chinese accent",
    "elderly",
    "female",
    "high pitch",
    "indian accent",
    "japanese accent",
    "korean accent",
    "low pitch",
    "male",
    "middle-aged",
    "moderate pitch",
    "portuguese accent",
    "russian accent",
    "teenager",
    "very high pitch",
    "very low pitch",
    "whisper",
    "young adult",
}


def validate_instruct(description: str) -> list[str]:
    """Return the unsupported items in a voice-design string."""
    items = [part.strip().lower() for part in description.split(",") if part.strip()]
    return [item for item in items if item not in INSTRUCT_VOCAB]


#: Always available, needs no reference audio or extra downloads.
BUILTIN = Voice(
    id="default",
    label="Default (voice design, male, moderate pitch)",
    language="ar",
    description="male, young adult, moderate pitch",
    tags=["design", "ar"],
)


class VoiceRegistry:
    def __init__(self, voices_dir: Path | None = None) -> None:
        self.dir = Path(voices_dir or get_settings().voices_dir)
        self._voices: dict[str, Voice] = {}
        self.load()

    def load(self) -> None:
        self._voices = {BUILTIN.id: BUILTIN}
        manifest = self.dir / MANIFEST_NAME
        if not manifest.exists():
            log.info("No voice manifest at %s; using built-in design voice only", manifest)
            return

        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read %s: %s", manifest, exc)
            return

        for entry in raw.get("voices", []):
            try:
                voice = self._build(entry)
            except (KeyError, ValueError) as exc:
                log.warning("Skipping voice entry %r: %s", entry.get("id", "?"), exc)
                continue
            self._voices[voice.id] = voice
        log.info("Loaded %d voices from %s", len(self._voices), manifest)

    def _build(self, entry: dict) -> Voice:
        voice_id = entry["id"]
        ref_audio = entry.get("ref_audio")
        if ref_audio:
            path = (self.dir / ref_audio).resolve()
            # Keep references inside the voices directory: the id comes from a
            # request parameter, so a traversal here would be a file-read primitive.
            if not path.is_relative_to(self.dir.resolve()):
                raise ValueError("ref_audio escapes the voices directory")
            if not path.exists():
                raise ValueError(f"ref_audio not found: {path}")
            ref_audio = str(path)
            if not entry.get("ref_text"):
                raise ValueError("cloning voices need ref_text (transcript of the reference)")
        elif not entry.get("description"):
            raise ValueError("voice needs either ref_audio+ref_text or a description")
        else:
            unsupported = validate_instruct(entry["description"])
            if unsupported:
                raise ValueError(
                    f"unsupported voice-design items {unsupported}. "
                    f"Use a comma-separated subset of: {', '.join(sorted(INSTRUCT_VOCAB))}"
                )

        return Voice(
            id=voice_id,
            label=entry.get("label", voice_id),
            language=entry.get("language", "ar"),
            ref_audio=ref_audio,
            ref_text=entry.get("ref_text"),
            description=entry.get("description"),
            tags=entry.get("tags", []),
        )

    def get(self, voice_id: str | None) -> Voice:
        if not voice_id:
            return self._voices[BUILTIN.id]
        try:
            return self._voices[voice_id]
        except KeyError:
            raise KeyError(
                f"Unknown voice {voice_id!r}. Available: {', '.join(sorted(self._voices))}"
            ) from None

    def list(self) -> list[Voice]:
        return list(self._voices.values())

    def validate_reference(self, voice: Voice) -> list[str]:
        """Warn about reference clips that will produce poor cloning."""
        if not voice.is_clone:
            return []
        from ..common.audio import load_audio_file
        from ..common.protocols import SAMPLE_RATE_IN

        problems: list[str] = []
        try:
            audio = load_audio_file(voice.ref_audio)
        except Exception as exc:
            return [f"could not read reference audio: {exc}"]

        seconds = len(audio) / SAMPLE_RATE_IN
        if seconds < MIN_REF_SECONDS:
            problems.append(f"reference is {seconds:.1f}s, want >= {MIN_REF_SECONDS}s")
        if seconds > MAX_REF_SECONDS:
            problems.append(f"reference is {seconds:.1f}s, want <= {MAX_REF_SECONDS}s")
        return problems
