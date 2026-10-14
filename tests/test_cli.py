"""Argument parsing, rendering and the exit statuses."""

import json
from pathlib import Path

import pytest

from edarecover.cli import COMMANDS, build_parser, main, quick
from edarecover.config import DEFAULT


def test_every_subcommand_is_reachable_from_the_parser():
    parser = build_parser()
    for name in COMMANDS:
        args = parser.parse_args([name] + (["corpus"] if name == "dalia" else []))
        assert args.handler is COMMANDS[name][0]
        assert args.renderer is COMMANDS[name][1]


def test_a_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_quick_mode_shrinks_the_sweep():
    reduced = quick(DEFAULT)
    assert len(reduced.blur_levels) < len(DEFAULT.blur_levels)
    assert reduced.n_seeds < DEFAULT.n_seeds
    assert reduced.fs == DEFAULT.fs


def test_quick_calibrate_prints_a_summary(capsys: pytest.CaptureFixture[str]):
    assert main(["--quick", "calibrate", "--noise-trials", "5"]) == 0
    out = capsys.readouterr().out
    assert "threshold" in out
    assert "coherence floor" in out


def test_json_output_parses(capsys: pytest.CaptureFixture[str]):
    assert main(["--quick", "--json", "calibrate", "--noise-trials", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"] == 8
    assert 0.0 <= payload["threshold"] <= 1.0
    assert set(payload) >= {"auc", "sensitivity", "specificity", "coherence_floor"}


def test_validate_without_a_cache_explains_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = main(["--cache", str(tmp_path / "absent.npz"), "validate"])
    assert code == 2
    assert "calibrate" in capsys.readouterr().err


def test_dalia_without_files_explains_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["dalia", str(tmp_path)]) == 2
    assert "no subject files" in capsys.readouterr().err


def test_dalia_reports_each_subject(corpus: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["--json", "dalia", str(corpus)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["subject"] for row in payload["subjects"]] == ["S1", "S2"]
    assert all(row["windows"] > 0 for row in payload["subjects"])
