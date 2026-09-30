"""Exercise the actual CLI and its artifact contract."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.reference_fixed_sampler import ROOT, UPSTREAM_COMMIT


def test_cli_runs_cpu_cases_and_records_results(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/smoke_local.py"),
         "--config", str(ROOT / "configs/local_smoke.yaml"),
         "--device", "cpu", "--output-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "local_smoke.json").read_text())
    assert report["status"] == "passed"
    assert report["scope"] == "tensor_only_fixed_sampler"
    assert report["device"] == "cpu"
    assert report["upstream_commit"] == UPSTREAM_COMMIT
    assert report["peak_gpu_memory_mb"] is None
    assert report["counts"]["passed"] >= 37
    assert report["counts"]["failed"] == report["counts"]["errors"] == 0
    assert len(report["cases"]) == report["counts"]["passed"]
    assert report["runtime_sec"] > 0


def test_missing_config_stops_before_running_tests(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/smoke_local.py"),
         "--config", str(tmp_path / "absent.yaml"),
         "--output-dir", str(tmp_path / "output")],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not (tmp_path / "output/local_smoke.junit.xml").exists()
    assert "missing" in result.stderr.lower()


def test_config_cannot_report_a_different_reference_checkout(tmp_path):
    config = tmp_path / "alternate.yaml"
    config.write_text(f"upstream_path: {tmp_path}\nupstream_commit: {UPSTREAM_COMMIT}\noutput_dir: artifacts\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/smoke_local.py"), "--config", str(config),
         "--output-dir", str(tmp_path / "output")],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "upstream_path must use the pinned submodule" in result.stderr
    assert not (tmp_path / "output/local_smoke.junit.xml").exists()


@pytest.mark.parametrize("missing", ["adaptive_noise/sampling_kernel.py", "tests/reference_fixed_sampler.py"])
def test_missing_required_code_stops_in_preflight(tmp_path, monkeypatch, capsys, missing):
    from scripts import smoke_local
    monkeypatch.setattr(smoke_local, "ROOT", tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(f"upstream_path: latent_grpo_minfix/Latent-GRPO-final\nupstream_commit: {UPSTREAM_COMMIT}\noutput_dir: artifacts\n")
    for relative in (
        "latent_grpo_minfix/Latent-GRPO-final/" + smoke_local.SAMPLER_RELATIVE,
        "tests/test_fixed_mode_identity.py", "tests/reference_fixed_sampler.py",
        "adaptive_noise/sampling_kernel.py",
    ):
        if relative != missing:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Controlled fixture; must not execute.\n")
    monkeypatch.setattr(sys, "argv", ["smoke_local.py", "--config", str(config)])
    assert smoke_local.main() == 2
    stderr = capsys.readouterr().err
    assert "missing required input" in stderr and missing in stderr
    assert not (tmp_path / "artifacts/local_smoke.junit.xml").exists()
