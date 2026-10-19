"""ditch backend: spec parsing, path promotion and headless argv."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tal.backends.ditch import DitchBackend
from tal.errors import SpecError
from tal.spec import load_spec
from tal.util import CommandResult

SPEC = """\
version: 1
name: ditch-test
seed: 1
responsible_use_ack: true
model:
  id: Qwen3-0.6B
  local_path: {gguf}
  eval_path: {hf_dir}
  device: cpu
slices:
  - name: harmful_core
    kind: target
    source: {bad}
  - name: harmless_daily
    kind: preserve
    source: {good}
split: {{fit: 0.5, select: 0.25, report: 0.25, seed: 1}}
abliteration:
  backend: ditch
  n_trials: 2
  n_startup_trials: 1
  max_response_length: 8
  timeout_minutes: 5
evaluation:
  capability:
    tasks: []
audit:
  enabled: false
  backend: none
"""


class _Bundle:
    def __init__(self, root: Path):
        self.root = root

    def select_path(self, name: str) -> Path:
        return self.root / "select" / f"{name}.txt"

    def fit_path(self, side: str) -> Path:
        return self.root / "fit" / f"{side}.txt"


class DitchSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "model.gguf").write_bytes(b"GGUF")
        hf = self.root / "hf"
        hf.mkdir()
        (hf / "config.json").write_text("{}", encoding="utf-8")
        (hf / "model.safetensors").write_bytes(b"x")
        (self.root / "bad.txt").write_text("a\nb\n", encoding="utf-8")
        (self.root / "good.txt").write_text("c\nd\n", encoding="utf-8")
        self.spec_path = self.root / "spec.yaml"
        self.spec_path.write_text(
            SPEC.format(
                gguf=self.root / "model.gguf",
                hf_dir=hf,
                bad=self.root / "bad.txt",
                good=self.root / "good.txt",
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_gguf_with_eval_path(self):
        spec = load_spec(self.spec_path)
        self.assertEqual(spec.abliteration.backend, "ditch")
        self.assertTrue(spec.model.is_gguf)
        self.assertEqual(spec.model.eval_target(), str((self.root / "hf").resolve()))
        self.assertEqual(spec.to_dict()["model"]["eval_path"], str((self.root / "hf").resolve()))

    def test_heretic_rejects_gguf(self):
        text = self.spec_path.read_text(encoding="utf-8").replace("backend: ditch", "backend: heretic")
        self.spec_path.write_text(text, encoding="utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_spec(self.spec_path)
        self.assertIn("GGUF", str(ctx.exception))

    def test_ditch_requires_eval_path_for_gguf(self):
        text = self.spec_path.read_text(encoding="utf-8")
        text = "\n".join(line for line in text.splitlines() if "eval_path:" not in line)
        self.spec_path.write_text(text, encoding="utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_spec(self.spec_path)
        self.assertIn("eval_path", str(ctx.exception))


class DitchBackendTests(unittest.TestCase):
    def test_build_config_and_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.gguf").write_bytes(b"GGUF")
            hf = root / "hf"
            hf.mkdir()
            (hf / "config.json").write_text("{}", encoding="utf-8")
            (hf / "model.safetensors").write_bytes(b"x")
            (root / "bad.txt").write_text("a\nb\n", encoding="utf-8")
            (root / "good.txt").write_text("c\nd\n", encoding="utf-8")
            spec_path = root / "spec.yaml"
            spec_path.write_text(
                SPEC.format(
                    gguf=root / "model.gguf",
                    hf_dir=hf,
                    bad=root / "bad.txt",
                    good=root / "good.txt",
                ),
                encoding="utf-8",
            )
            spec = load_spec(spec_path)
            workdir = root / "work"
            workdir.mkdir()
            bundle = _Bundle(root)
            (root / "select").mkdir()
            (root / "fit").mkdir()
            for name in ("harmful_core", "harmless_daily"):
                (root / "select" / f"{name}.txt").write_text("x\n", encoding="utf-8")
            (root / "fit" / "bad.txt").write_text("x\n", encoding="utf-8")
            (root / "fit" / "good.txt").write_text("x\n", encoding="utf-8")

            backend = DitchBackend(executable=str(root / "ditch.exe"))
            (root / "ditch.exe").write_bytes(b"!")
            config_text, paths = backend.build_config(spec, bundle, workdir)
            self.assertIn("KeywordRate", config_text)
            self.assertIn(str(root / "select" / "harmful_core.txt"), config_text)
            self.assertIn("KLDivergence", config_text)

            seen = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                raw_out = Path(paths["raw_out"])
                raw_out.mkdir(parents=True, exist_ok=True)
                (raw_out / "config.json").write_text("{}", encoding="utf-8")
                (raw_out / "model.safetensors").write_bytes(b"x")
                return CommandResult(
                    argv=command,
                    returncode=0,
                    duration_s=1.0,
                    log_path=workdir / "ditch.log",
                    tail="ok",
                )

            original = __import__("tal.backends.ditch", fromlist=["run_command"]).run_command
            __import__("tal.backends.ditch", fromlist=["run_command"]).run_command = fake_run
            try:
                result = backend.run(spec, bundle, workdir, _Logger())
            finally:
                __import__("tal.backends.ditch", fromlist=["run_command"]).run_command = original

            command = seen["command"]
            self.assertIn("--no-input", command)
            self.assertIn("--trial-index", command)
            self.assertIn("--export-format", command)
            self.assertIn("hf", command)
            self.assertTrue(result.model_path.exists())
            self.assertTrue((result.model_path / "config.json").exists())
            self.assertEqual(result.metadata["input_format"], "gguf")


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass


if __name__ == "__main__":
    unittest.main()

