"""voicegw CLI."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .common.config import Profile, Settings, get_settings, resolve_profile

app = typer.Typer(help="Arabic voice gateway: STT + TTS over REST, WebSocket and MCP.")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        # Model downloads emit one INFO line per HTTP request, which buries
        # the messages that actually matter.
        for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def serve(
    host: str | None = None,
    port: int | None = None,
    profile: Profile | None = typer.Option(None, help="Override VOICEGW_PROFILE."),
    stt: str | None = typer.Option(None, "--stt", help="Override VOICEGW_ASR_ENGINE (e.g. cohere-asr-cpu, faster-whisper)."),
    tts: str | None = typer.Option(None, "--tts", help="Override VOICEGW_TTS_ENGINE (e.g. omnivoice-cpu, omnivoice-gguf, piper)."),
    reload: bool = False,
    verbose: bool = False,
) -> None:
    """Run the REST + WebSocket gateway."""
    import os

    import uvicorn

    _setup_logging(verbose)
    if profile is not None:
        os.environ["VOICEGW_PROFILE"] = profile.value
        # If user explicitly requests a profile, don't inherit lingering session pins from previous runs
        if stt is None and "VOICEGW_ASR_ENGINE" in os.environ:
            del os.environ["VOICEGW_ASR_ENGINE"]
        if tts is None and "VOICEGW_TTS_ENGINE" in os.environ:
            del os.environ["VOICEGW_TTS_ENGINE"]
        get_settings.cache_clear()
    if stt is not None:
        os.environ["VOICEGW_ASR_ENGINE"] = stt
        get_settings.cache_clear()
    if tts is not None:
        os.environ["VOICEGW_TTS_ENGINE"] = tts
        get_settings.cache_clear()

    settings = get_settings()
    bind = host or settings.host
    if bind not in {"127.0.0.1", "localhost", "::1"}:
        # There is no authentication anywhere in the gateway: binding it to a
        # routable address hands anyone on the network unmetered use of the
        # GPU, plus whatever /healthz reveals about the machine.
        console.print(
            f"[yellow]WARNING[/] binding to {bind}, not loopback. The gateway has no "
            "authentication — anyone who can reach this port can run transcription "
            "and synthesis on your hardware. Put it behind a reverse proxy that "
            "handles auth, or bind to 127.0.0.1."
        )
    uvicorn.run(
        "voicegw.api.app:app",
        host=bind,
        port=port or settings.port,
        reload=reload,
        log_level="debug" if verbose else "info",
    )


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio (the gateway must already be running)."""
    from .mcp.server import main

    main()


@app.command()
def fetch(
    include_gated: bool = typer.Option(
        True, help="Also fetch the gated Cohere ASR repo (needs HF_TOKEN)."
    ),
    force: bool = typer.Option(False, help="Re-download even if already cached."),
    verbose: bool = False,
) -> None:
    """Download every model into the local cache so the gateway can run offline.

    Run this once while you have a network connection, then set
    VOICEGW_OFFLINE=1.
    """
    import os

    from huggingface_hub import snapshot_download

    from .common import offline as offline_mod

    _setup_logging(verbose)
    # Never run the fetcher in offline mode -- it would have nothing to do.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)

    settings = get_settings()
    token = settings.hf_token or os.environ.get("HF_TOKEN")
    targets = list(offline_mod.assets(settings))

    failures = 0
    for asset in targets:
        if asset.gated and not include_gated:
            console.print(f"[dim]skip[/] {asset.repo_id} (gated)")
            continue
        if not force and offline_mod.is_cached(asset.repo_id):
            console.print(f"[green]cached[/] {asset.repo_id}")
            continue
        if asset.gated and not token:
            console.print(
                f"[red]skip[/] {asset.repo_id} — gated repo and HF_TOKEN is not set. "
                "Accept its terms on the model page, then set HF_TOKEN in .env."
            )
            failures += 1
            continue

        console.print(f"[cyan]fetching[/] {asset.repo_id} ({asset.size_hint}) — {asset.purpose}")
        try:
            snapshot_download(asset.repo_id, token=token if asset.gated else None)
        except Exception as exc:
            from .common.errors import hint_for

            console.print(f"[red]failed[/] {asset.repo_id}: {exc}")
            hint = hint_for(exc)
            if hint:
                console.print(f"[yellow]{hint}[/]")
            failures += 1
            continue
        console.print(f"[green]done[/] {asset.repo_id}")

    if offline_mod.needs_external_audio_tokenizer():
        repo = offline_mod.AUDIO_TOKENIZER_REPO
        console.print(f"[cyan]fetching[/] {repo} (OmniVoice tokenizer is not bundled)")
        try:
            snapshot_download(repo)
        except Exception as exc:
            console.print(f"[red]failed[/] {repo}: {exc}")
            failures += 1

    # Piper keeps its voices outside the HF cache.
    from .engines import piper_tts

    if piper_tts.is_cached():
        console.print(f"[green]cached[/] piper/{piper_tts.DEFAULT_VOICE_MODEL}")
    else:
        console.print(f"[cyan]fetching[/] piper/{piper_tts.DEFAULT_VOICE_MODEL} (~60 MB)")
        try:
            from piper.download_voices import download_voice

            target = piper_tts.voices_dir()
            target.mkdir(parents=True, exist_ok=True)
            download_voice(piper_tts.DEFAULT_VOICE_MODEL, target)
            console.print(f"[green]done[/] piper/{piper_tts.DEFAULT_VOICE_MODEL}")
        except Exception as exc:
            # Piper is only a fallback engine, so this is not fatal.
            console.print(f"[yellow]skip[/] piper voice: {exc}")

    console.print(f"\ncache: [dim]{offline_mod.cache_dir()}[/]")
    if failures:
        console.print(f"[red]{failures} model(s) unavailable[/] — offline mode will be degraded.")
        raise typer.Exit(1)
    console.print("[green]All models cached.[/] Set VOICEGW_OFFLINE=1 to run without a network.")


def _run(coro) -> None:
    """Run a local command, reporting model-load failures as advice.

    These commands skip warmup, so a bad environment surfaces as a raw
    exception from deep inside transformers. Translate it.
    """
    from .common.errors import EngineUnavailable, hint_for

    try:
        asyncio.run(coro)
    except EngineUnavailable as exc:
        console.print(f"[red]{exc.reason}[/]")
        if exc.hint:
            console.print(f"[yellow]{exc.hint}[/]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}: {exc}[/]")
        hint = hint_for(exc)
        if hint:
            console.print(f"[yellow]{hint}[/]")
        console.print("[dim]Run `voicegw doctor` to check your environment.[/]")
        raise typer.Exit(1) from exc


@app.command()
def doctor(profile: Profile | None = None) -> None:
    """Check the environment: profile resolution, deps, token, voices."""
    import os

    _setup_logging(False)
    if profile is not None:
        os.environ["VOICEGW_PROFILE"] = profile.value
        get_settings.cache_clear()

    settings = get_settings()
    table = Table(title="voicegw doctor", show_lines=False)
    table.add_column("check")
    table.add_column("result")

    # The table is printed from a `finally` below. Importing torch and
    # transformers takes several seconds cold, and whatever goes wrong in
    # there must not swallow the rows already gathered -- a diagnostic that
    # reports nothing when the environment is broken is worse than useless.
    try:
        _doctor_checks(settings, table)
    finally:
        console.print(table)


#: Third-party packages `doctor` reports on, in import order.
_DEPENDENCIES = (
    "torch",
    "transformers",
    "omnivoice",
    "faster_whisper",
    "silero_vad",
    "piper",
    "bitsandbytes",
)


def _probe_dependencies(table: Table) -> None:
    """Import each dependency and record what happened.

    Importing arbitrary third-party code is hostile territory: a package can
    call ``sys.exit()`` when it dislikes the environment, which raises
    ``SystemExit`` -- a ``BaseException``, so a plain ``except Exception``
    would let it terminate the process. Naming the culprit is the whole point
    of this command, so nothing here may kill it silently.
    """
    for mod in _DEPENDENCIES:
        try:
            __import__(mod)
            table.add_row(mod, "[green]installed[/]")
        except ImportError:
            table.add_row(mod, "[yellow]missing[/]")
        except KeyboardInterrupt:
            table.add_row(mod, "[yellow]interrupted[/]")
            raise  # the user asked to stop; the table still prints
        except BaseException as exc:
            table.add_row(mod, f"[red]broken: {type(exc).__name__}: {exc}[/]")


def _doctor_checks(settings: Settings, table: Table) -> None:
    import os

    try:
        res = resolve_profile(settings)
        table.add_row("profile", f"[green]{res.profile.value}[/] ({res.reason})")
        table.add_row("device", res.device)
    except RuntimeError as exc:
        table.add_row("profile", f"[red]{exc}[/]")
        res = None

    console.print("[dim]importing dependencies (cold start takes a few seconds)...[/]")
    _probe_dependencies(table)

    token = settings.hf_token or os.environ.get("HF_TOKEN")
    table.add_row(
        "HF_TOKEN",
        "[green]set[/]"
        if token
        else "[red]missing[/] — the Cohere ASR repo is gated; accept its terms and set a token",
    )

    voices_dir = Path(settings.voices_dir)
    table.add_row(
        "voices",
        f"[green]{voices_dir}/manifest.json[/]"
        if (voices_dir / "manifest.json").exists()
        else "[yellow]no manifest — built-in design voice only[/]",
    )

    if res is not None:
        from .engines import registry

        stt = settings.asr_engine or registry.PROFILE_DEFAULTS[res.profile]["stt"]
        tts = settings.tts_engine or registry.PROFILE_DEFAULTS[res.profile]["tts"]
        table.add_row("stt engine", stt)
        table.add_row("tts engine", tts)

        if stt == "cohere-asr" and res.profile is Profile.GPU:
            from .engines.cohere_asr import VRAM_MIB, resolve_quantization

            quant = resolve_quantization(settings)
            note = f"{quant.value} (~{VRAM_MIB[quant]} MiB)"
            if settings.asr_quantization.value == "auto":
                note += " — auto"
            table.add_row("asr quantization", note)

    from .common import offline as offline_mod

    table.add_row(
        "offline mode",
        "[green]on[/] — models load from the local cache only"
        if settings.offline
        else "[dim]off[/] — missing weights will be downloaded",
    )
    table.add_row("model cache", str(offline_mod.cache_dir()))
    for asset in offline_mod.assets(settings):
        if offline_mod.is_cached(asset.repo_id):
            state = "[green]cached[/]"
        elif settings.offline:
            state = f"[red]NOT cached[/] — run `voicegw fetch` online ({asset.size_hint})"
        else:
            state = f"[yellow]not cached[/] — will download on first use ({asset.size_hint})"
        table.add_row(f"  {asset.key}", state)


@app.command()
def transcribe(
    path: Path = typer.Argument(..., exists=True, readable=True),
    language: str = "ar",
    profile: Profile | None = None,
    verbose: bool = False,
) -> None:
    """Transcribe an audio file locally (no server)."""
    import os

    _setup_logging(verbose)
    if profile is not None:
        os.environ["VOICEGW_PROFILE"] = profile.value
        get_settings.cache_clear()

    from .core import VoiceCore

    async def run() -> None:
        core = VoiceCore()
        await core.startup(warmup=False)
        result = await core.transcribe_bytes(path.read_bytes(), language)
        console.print(f"[dim]{result.engine} · {result.duration_s:.1f}s[/]")
        console.print(result.text or "[yellow](no speech detected)[/]")
        await core.shutdown()

    _run(run())


@app.command()
def describe(
    path: Path = typer.Argument(..., exists=True, readable=True),
    verbose: bool = False,
) -> None:
    """Analyze acoustic prosody of an audio file and emit OmniVoice prompt tags."""
    _setup_logging(verbose)
    from .common.audio import load_audio_file
    from .engines.prosody import describe_voice

    audio = load_audio_file(str(path))
    description, metrics = describe_voice(audio, 16000)
    console.print(f"[bold cyan]OmniVoice Description:[/] [green]{description}[/]")
    console.print("[dim]Acoustic Metrics:[/]")
    for k, v in metrics.items():
        console.print(f"  {k}: [bold]{v}[/]")



@app.command()
def say(
    text: str,
    out: Path = Path("out.wav"),
    voice: str = "default",
    profile: Profile | None = None,
    verbose: bool = False,
) -> None:
    """Synthesize text to a WAV file locally (no server)."""
    import os

    _setup_logging(verbose)
    if profile is not None:
        os.environ["VOICEGW_PROFILE"] = profile.value
        get_settings.cache_clear()

    import numpy as np

    from .common.audio import concat_with_crossfade, encode_wav
    from .core import VoiceCore

    async def run() -> None:
        core = VoiceCore()
        await core.startup(warmup=False)
        chunks = [c async for c in core.synthesize(text, voice)]
        if not chunks:
            console.print("[red]no audio produced[/]")
            return
        sr = chunks[0].sample_rate
        fade_samples = int(sr * 0.035)
        samples = concat_with_crossfade([c.samples for c in chunks], fade_samples)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(encode_wav(samples, sr))
        console.print(f"[green]wrote[/] {out} ({len(samples) / sr:.1f}s)")
        await core.shutdown()

    _run(run())


if __name__ == "__main__":
    app()
