"""`doctor` must survive the environment it is diagnosing.

Its whole purpose is to run on a broken machine and say what is broken, so a
failure anywhere in the middle must still leave the user with the rows already
gathered -- otherwise the symptom is a silent exit that reveals nothing.
"""

import builtins

import pytest
from rich.console import Console
from rich.table import Table

from voicegw import cli


@pytest.fixture
def captured(monkeypatch):
    """Route the CLI's console into a buffer we can assert on."""
    console = Console(record=True, width=200, no_color=True)
    monkeypatch.setattr(cli, "console", console)
    return console


def _fake_import(name: str, exc: BaseException):
    """Replace `__import__` so one module name fails in a chosen way."""
    real = builtins.__import__

    def stub(mod, *args, **kwargs):
        if mod == name:
            raise exc
        return real(mod, *args, **kwargs)

    return stub


def test_a_dependency_that_hard_exits_is_reported_not_fatal(monkeypatch, captured):
    """`sys.exit()` during an import raises SystemExit, not Exception.

    A plain `except Exception` lets it through and kills the process with no
    output at all -- the exact failure this command exists to prevent.
    """
    monkeypatch.setattr(cli, "_DEPENDENCIES", ("torch", "wedged"))
    monkeypatch.setattr(builtins, "__import__", _fake_import("wedged", SystemExit(1)))

    table = Table()
    table.add_column("check")
    table.add_column("result")
    cli._probe_dependencies(table)  # must not raise

    captured.print(table)
    out = captured.export_text()
    assert "wedged" in out
    assert "SystemExit" in out
    assert "torch" in out  # the healthy probe is still reported


def test_a_broken_dependency_does_not_hide_the_ones_after_it(monkeypatch, captured):
    monkeypatch.setattr(cli, "_DEPENDENCIES", ("wedged", "torch"))
    monkeypatch.setattr(builtins, "__import__", _fake_import("wedged", RuntimeError("bad ABI")))

    table = Table()
    table.add_column("check")
    table.add_column("result")
    cli._probe_dependencies(table)

    captured.print(table)
    out = captured.export_text()
    assert "bad ABI" in out
    assert "torch" in out


def test_missing_and_broken_are_told_apart(monkeypatch, captured):
    """Not installed and installed-but-unusable need different fixes."""
    monkeypatch.setattr(cli, "_DEPENDENCIES", ("absent",))
    monkeypatch.setattr(builtins, "__import__", _fake_import("absent", ImportError("no module")))

    table = Table()
    table.add_column("check")
    table.add_column("result")
    cli._probe_dependencies(table)

    captured.print(table)
    assert "missing" in captured.export_text()


def test_interrupt_stops_the_probe_but_still_records_it(monkeypatch, captured):
    monkeypatch.setattr(cli, "_DEPENDENCIES", ("wedged", "torch"))
    monkeypatch.setattr(builtins, "__import__", _fake_import("wedged", KeyboardInterrupt()))

    table = Table()
    table.add_column("check")
    table.add_column("result")
    with pytest.raises(KeyboardInterrupt):
        cli._probe_dependencies(table)

    captured.print(table)
    assert "interrupted" in captured.export_text()


def test_doctor_prints_what_it_learned_even_when_a_check_explodes(monkeypatch, captured):
    """The table is buffered until the end, so it must print from a `finally`."""

    def explode(settings, table):
        table.add_row("profile", "gpu")
        raise SystemExit(1)

    monkeypatch.setattr(cli, "_doctor_checks", explode)

    with pytest.raises(SystemExit):
        cli.doctor()

    out = captured.export_text()
    assert "voicegw doctor" in out
    assert "gpu" in out
