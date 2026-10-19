import tempfile
import unittest
from pathlib import Path

from tal.demo import build_demo
from tal.errors import SpecError
from tal.spec import dump_effective_spec, load_spec


class SpecTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = build_demo(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_demo_spec_loads(self):
        spec = load_spec(self.path)
        self.assertEqual(spec.name, "tal-demo-dry-run")
        self.assertEqual(spec.model.id, "Qwen/Qwen3-0.6B")
        self.assertEqual([s.name for s in spec.slices_of("target")], ["harmful_core"])
        self.assertEqual([s.name for s in spec.slices_of("preserve")], ["harmless_daily"])
        self.assertEqual(spec.abliteration.backend, "simulate")
        self.assertAlmostEqual(spec.gates.target_min_drop, 0.30)
        self.assertTrue(spec.evaluation.kl_enabled)
        self.assertEqual(spec.report.language, "zh")
        self.assertAlmostEqual(spec.split.fit + spec.split.select + spec.split.report, 1.0)

    def test_fingerprint_is_stable_and_stage_scoped(self):
        a = load_spec(self.path)
        b = load_spec(self.path)
        self.assertEqual(a.fingerprint("prepare", dry_run=True), b.fingerprint("prepare", dry_run=True))
        self.assertEqual(a.fingerprint("abliterate", dry_run=True), b.fingerprint("abliterate", dry_run=True))
        # Changing a gate must not invalidate the expensive abliteration stage.
        c = load_spec(self.path, ["gates.target.min_absolute_drop=0.95"])
        self.assertEqual(c.fingerprint("abliterate", dry_run=True), a.fingerprint("abliterate", dry_run=True))
        self.assertNotEqual(c.fingerprint("gates", dry_run=True), a.fingerprint("gates", dry_run=True))

    def test_override_nested_and_capability(self):
        spec = load_spec(self.path, ["evaluation.capability.tasks.0.name=renamed", "seed=99"])
        self.assertEqual(spec.evaluation.capability_tasks[0].name, "renamed")
        self.assertEqual(spec.seed, 99)

    def test_effective_snapshot_round_trips(self):
        """spec.effective.yaml is reloaded by `tal resume` / `tal report`; every
        value (capability tasks, KL limits, objective weights) must survive."""
        spec = load_spec(self.path)
        effective = self.tmp / "effective.yaml"
        effective.write_text(dump_effective_spec(spec), encoding="utf-8")
        reloaded = load_spec(effective)
        self.assertEqual(spec.to_dict(), reloaded.to_dict())
        self.assertEqual(len(reloaded.evaluation.capability_tasks), 1)
        self.assertEqual(reloaded.evaluation.capability_tasks[0].type, "mcq")
        self.assertEqual(reloaded.evaluation.kl_max_prompts, spec.evaluation.kl_max_prompts)

    def test_validation_rejects_unknown_top_level_key(self):
        # A misplaced `capability:` block would otherwise be silently ignored.
        with self.assertRaisesRegex(SpecError, "unknown top-level key"):
            load_spec(self.path, ["capability.tasks=[]"])

    def test_validation_rejects_bad_split(self):
        with self.assertRaisesRegex(SpecError, "must equal 1.0"):
            load_spec(self.path, ["split.fit=0.9"])

    def test_validation_requires_responsible_use_ack(self):
        with self.assertRaisesRegex(SpecError, "responsible_use_ack"):
            load_spec(self.path, ["responsible_use_ack=false"])

    def test_validation_rejects_duplicate_slice_names(self):
        with self.assertRaisesRegex(SpecError, "duplicate"):
            load_spec(self.path, ["slices.1.name=harmful_core"])

    def test_validation_rejects_missing_source(self):
        with self.assertRaisesRegex(SpecError, "not found"):
            load_spec(self.path, ["slices.0.source=missing.txt"])


if __name__ == "__main__":
    unittest.main()
