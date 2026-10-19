"""Audit wiring tests: the Senbonzakura track/compass contract, without running it."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tal import audit as audit_mod
from tal.audit import _track_split_sizes, run_audit
from tal.spec import AuditSpec
from tal.util import Logger


class TrackSplitTests(unittest.TestCase):
    def test_small_corpus_keeps_measure_non_empty(self) -> None:
        sizes = _track_split_sizes(6, 6)
        self.assertIsNotNone(sizes)
        assert sizes is not None
        self.assertGreaterEqual(6 - sizes["fit"] - sizes["search"], 1)

    def test_too_small_returns_none(self) -> None:
        self.assertIsNone(_track_split_sizes(2, 2))


class AuditCommandTests(unittest.TestCase):
    def test_track_and_compass_flags_follow_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            for split in ("fit", "select", "report"):
                (run_dir / "data" / split).mkdir(parents=True)
                (run_dir / "data" / split / "harmful_core.txt").write_text(
                    f"{split} bad 1\n{split} bad 2\n", encoding="utf-8"
                )
                (run_dir / "data" / split / "harmless_daily.txt").write_text(
                    f"{split} good 1\n{split} good 2\n", encoding="utf-8"
                )
            model = root / "model"
            model.mkdir()

            calls: list[list[str]] = []

            def fake_run(argv, **kwargs):
                calls.append(list(argv))
                if argv[1] == "track":
                    track_dir = Path(argv[argv.index("--out") + 1])
                    track_dir.mkdir(parents=True, exist_ok=True)
                    (track_dir / "track.json").write_text(
                        json.dumps(
                            {
                                "schema": "senbonzakura-track/1",
                                "counts": {
                                    "harmful": {"fit": 3, "search": 1, "measure": 2},
                                    "harmless": {"fit": 3, "search": 1, "measure": 2},
                                },
                                "skip_harmful": 1,
                                "skip_harmless": 1,
                                "n_harmful": 2,
                                "n_harmless": 2,
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    out = Path(argv[argv.index("--out") + 1])
                    out.write_text(json.dumps({"auc": 0.91}), encoding="utf-8")
                return SimpleNamespace(returncode=0, tail="", duration_s=0.0)

            spec = SimpleNamespace(
                audit=AuditSpec(enabled=True),
                abliteration=SimpleNamespace(backend="heretic"),
                slices_of=lambda kind: (
                    [SimpleNamespace(name="harmful_core")]
                    if kind == "target"
                    else [SimpleNamespace(name="harmless_daily")]
                ),
            )
            logger = Logger(root / "tal.log", quiet=True)
            with mock.patch.object(audit_mod, "which", return_value="senbonzakura"), mock.patch.object(
                audit_mod, "run_command", side_effect=fake_run
            ):
                payload = run_audit(spec, run_dir, model, logger)

            self.assertEqual(payload["status"], "ok")
            self.assertAlmostEqual(payload["auc"], 0.91)
            track_cmd = calls[0]
            self.assertIn("--fit", track_cmd)
            self.assertIn("--search", track_cmd)
            compass_cmd = calls[1]
            self.assertEqual(compass_cmd[compass_cmd.index("--n") + 1], "2")
            self.assertEqual(compass_cmd[compass_cmd.index("--skip-harmful") + 1], "1")
            self.assertEqual(compass_cmd[compass_cmd.index("--skip-harmless") + 1], "1")
            self.assertEqual(payload["track"]["compass_n"], 2)


if __name__ == "__main__":
    unittest.main()
