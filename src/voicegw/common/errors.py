"""Errors the façades translate into meaningful HTTP / WebSocket responses."""

from __future__ import annotations


class EngineUnavailable(RuntimeError):
    """An engine failed to load, so the capability it provides is offline.

    Carries the original failure and a remediation hint, because the usual
    cause -- a gated Hugging Face repo with no token -- is fixable by the user
    in under a minute if we actually tell them what to do.
    """

    def __init__(self, engine_id: str, reason: str, hint: str = "") -> None:
        self.engine_id = engine_id
        self.reason = reason
        self.hint = hint
        message = f"{engine_id} is unavailable: {reason}"
        if hint:
            message = f"{message} -- {hint}"
        super().__init__(message)


#: Matched against the load failure to attach a remediation hint.
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("gated repo", "401 client error", "access to model", "must have access"),
        "Accept the model terms on its Hugging Face page, then set HF_TOKEN in .env.",
    ),
    (
        ("out of memory", "cuda oom"),
        "Not enough VRAM. Try VOICEGW_RESIDENCY=exclusive, a smaller model, "
        "or VOICEGW_PROFILE=cpu.",
    ),
    (
        ("no module named", "importerror", "winerror 127"),
        "A dependency is missing or ABI-mismatched. Re-run the install for your "
        "profile and check `voicegw doctor`.",
    ),
    # Checked before the generic network hint: an offline cache miss also
    # mentions disabled outgoing traffic, and the specific advice is better.
    (
        (
            "localentrynotfound",
            "offlinemode",
            "offline mode is enabled",
            "outgoing traffic has been disabled",
            "cannot find the requested files in the disk cache",
            "local_files_only",
        ),
        "Offline mode is on but these weights are not cached. Run `voicegw fetch` "
        "once with a network connection, or unset VOICEGW_OFFLINE.",
    ),
    (
        ("connection", "timed out", "max retries", "nameresolution"),
        "The model could not be downloaded. Check your network, or pre-fetch the "
        "weights into the Hugging Face cache.",
    ),
)


def hint_for(exc: BaseException) -> str:
    """Best-effort remediation advice for a model-load failure."""
    # The exception type carries the signal for some failures (notably
    # huggingface_hub's LocalEntryNotFoundError), so match against it too.
    text = f"{type(exc).__name__}\n{exc}".lower()
    for needles, hint in _HINTS:
        if any(n in text for n in needles):
            return hint
    return ""
