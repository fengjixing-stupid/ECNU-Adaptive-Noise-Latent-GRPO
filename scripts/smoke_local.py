"""Run the fixed CPU sampler contracts and save an auditable JSON report."""

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[1]
SAMPLER_RELATIVE = "Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py"
SUITE = "tests/test_fixed_mode_identity.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_smoke.yaml")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if not args.config.is_file():
            raise FileNotFoundError(f"missing required config: {args.config}")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        upstream = ROOT / config["upstream_path"]
        if upstream.resolve() != (ROOT / "latent_grpo_minfix/Latent-GRPO-final").resolve():
            raise ValueError("upstream_path must use the pinned submodule read by the reference tests")
        sampler = upstream / SAMPLER_RELATIVE
        kernel = ROOT / "adaptive_noise/sampling_kernel.py"
        for path in (sampler, ROOT / SUITE, kernel, ROOT / "tests/reference_fixed_sampler.py"):
            if not path.is_file():
                raise FileNotFoundError(f"missing required input: {path}")
        actual_commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual_commit != config["upstream_commit"]:
            raise ValueError("upstream HEAD differs from the configured minfix commit")
        changed = subprocess.run(
            ["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--", SAMPLER_RELATIVE])
        if changed.returncode != 0:
            raise ValueError("upstream sampler differs from the pinned commit")
        versions = {name: importlib.metadata.version(name)
                    for name in ("torch", "pytest", "PyYAML")}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError,
            importlib.metadata.PackageNotFoundError, yaml.YAMLError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2

    output = args.output_dir or Path(config["output_dir"])
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    junit = output / "local_smoke.junit.xml"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "--tb=short",
         f"--junitxml={junit}"], cwd=ROOT, capture_output=True, text=True)
    (output / "local_smoke.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    cases = []
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    if junit.is_file():
        for case in ET.parse(junit).getroot().iter("testcase"):
            status = "passed"
            for tag, state in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped")):
                if case.find(tag) is not None:
                    status = state
                    break
            counts[status] += 1
            cases.append({"name": case.attrib["name"], "status": status,
                          "runtime_sec": float(case.attrib.get("time", 0))})
    passed = result.returncode == 0 and counts["passed"] > 0 and not any(
        counts[key] for key in ("failed", "errors", "skipped"))
    report = {
        "schema_version": 1, "scope": "tensor_only_fixed_sampler",
        "status": "passed" if passed else "failed", "device": args.device,
        "seeds": [0, 1, 2], "chunk_size": 8192, "top_k": 10,
        "upstream_commit": actual_commit,
        "upstream_sampler_sha256": sha256(sampler.read_bytes()).hexdigest(),
        "kernel_sha256": sha256(kernel.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "versions": versions,
        "peak_gpu_memory_mb": None, "runtime_sec": time.perf_counter() - started,
        "pytest_exit_code": result.returncode, "counts": counts, "cases": cases,
    }
    destination = output / "local_smoke.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {counts}; report={destination}")
    if not passed:
        print(f"STOP: inspect {output / 'local_smoke.log'}; no retry performed", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
