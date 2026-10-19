import tempfile
import tomllib
import unittest
from pathlib import Path

from tal.backends.heretic import HereticBackend
from tal.datasets import materialize_datasets
from tal.demo import build_demo
from tal.spec import load_spec


class HereticConfigTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.demo = build_demo(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, overrides=None):
        spec = load_spec(self.demo, ["abliteration.backend=heretic"] + (overrides or []))
        bundle = materialize_datasets(spec, self.tmp / "run")
        text, paths = HereticBackend(executable="heretic").build_config(spec, bundle, self.tmp / "work")
        return tomllib.loads(text), paths, bundle

    def test_generated_config_is_valid_toml_and_has_expected_tables(self):
        cfg, paths, bundle = self._build()

        self.assertEqual(cfg["model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(cfg["trial_index"], 0)
        self.assertEqual(cfg["model_action"], "save")
        self.assertEqual(cfg["export_strategy"], "merge")
        self.assertEqual(cfg["checkpoint_action"], "restart")
        self.assertEqual(cfg["save_directory"], str(paths["model_out"]))
        self.assertEqual(cfg["scorers"][0]["instance_name"], "targeted")
        self.assertTrue(cfg["scorers"][0]["plugin"].endswith("heretic_plugin.py:TargetedObjective"))
        self.assertEqual(cfg["scorers"][0]["optimization"], "minimize")
        self.assertEqual(cfg["modifiers"][0]["plugin"], "heretic.modifiers.ara.ARA")

        scorer = cfg["scorer"]["TargetedObjective_targeted"]
        self.assertEqual(scorer["refusal_weight"], 1)
        self.assertEqual(scorer["preserve_kl_weight"], 2)
        self.assertEqual([s["name"] for s in scorer["target_slices"]], ["harmful_core"])
        self.assertEqual([s["name"] for s in scorer["preserve_slices"]], ["harmless_daily"])
        self.assertEqual(scorer["target_slices"][0]["path"], str(bundle.select_path("harmful_core")))
        self.assertEqual(paths["model_out"].name, "model")

    def test_abliteration_modifier_selection(self):
        cfg, _, _ = self._build(["abliteration.modifier=abliteration"])
        self.assertEqual(cfg["modifiers"][0]["plugin"], "heretic.modifiers.abliteration.Abliteration")
        self.assertIn("Abliteration", cfg["modifier"])


if __name__ == "__main__":
    unittest.main()
