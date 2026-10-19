import json
import tempfile
import unittest
from pathlib import Path

from tal.demo import build_demo
from tal.errors import GateFailed, StageError
from tal.pipeline import open_run, run_experiment
from tal.spec import load_spec


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = build_demo(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def _prepare(self, overrides=None):
        spec = load_spec(self.path, overrides or [])
        ctx, state = open_run(spec, self.tmp / "runs", run_id="dry", dry_run=True)
        return ctx, state

    def test_dry_run_completes_and_writes_report(self):
        ctx, _ = self._prepare()
        result = run_experiment(ctx)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["gates_passed"])
        report = ctx.run_dir / "reports" / "report.md"
        self.assertTrue(report.exists())
        self.assertTrue((ctx.run_dir / "reports" / "report.json").exists())
        self.assertTrue((ctx.run_dir / "reports" / "model_card.md").exists())
        payload = json.loads((ctx.run_dir / "reports" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["verdict"], "通过（模拟）")
        self.assertTrue(payload["gates_passed"])
        # The report must never leak harmful prompt text.
        harmful = (self.tmp / "demo-data" / "target.txt").read_text(encoding="utf-8").splitlines()[0]
        self.assertNotIn(harmful, report.read_text(encoding="utf-8"))

    def test_second_run_is_fully_cached(self):
        ctx, state = self._prepare()
        run_experiment(ctx)
        attempts = {name: entry["attempts"] for name, entry in state.data["stages"].items()}

        result = run_experiment(ctx)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            {name: entry["attempts"] for name, entry in ctx.state.data["stages"].items()},
            attempts,
        )

    def test_gate_failure_raises_and_marks_run(self):
        ctx, _ = self._prepare(["gates.kl.max_mean=0.001"])
        with self.assertRaises(GateFailed):
            run_experiment(ctx)
        state = json.loads((ctx.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "gate_failed")
        gates = json.loads((ctx.run_dir / "stages" / "gates" / "gates.json").read_text(encoding="utf-8"))
        self.assertFalse(gates["passed"])

    def test_only_stage_requires_dependencies(self):
        ctx, _ = self._prepare()
        with self.assertRaises(StageError):
            run_experiment(ctx, only_stage="gates")


if __name__ == "__main__":
    unittest.main()
