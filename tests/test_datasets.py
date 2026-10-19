import json
import tempfile
import unittest
from pathlib import Path

from tal.datasets import load_bundle, load_capability_items, materialize_datasets
from tal.demo import build_demo
from tal.spec import load_spec


def _jsonl_ids(path):
    return {
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


class DatasetsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.spec = load_spec(build_demo(self.tmp))
        self.run_dir = self.tmp / "run"

    def tearDown(self):
        self._tmp.cleanup()

    def test_materialize_writes_split_files_and_disjoint_prompts(self):
        bundle = materialize_datasets(self.spec, self.run_dir)
        self.assertTrue((self.run_dir / "data" / "manifest.json").exists())
        self.assertTrue(bundle.fit_path("bad").exists())
        self.assertTrue(bundle.fit_path("good").exists())

        total = 0
        for slice_name in ("harmful_core", "harmless_daily"):
            ids = [
                _jsonl_ids(self.run_dir / "data" / "fit" / f"{slice_name}.jsonl"),
                _jsonl_ids(self.run_dir / "data" / "select" / f"{slice_name}.jsonl"),
                _jsonl_ids(self.run_dir / "data" / "report" / f"{slice_name}.jsonl"),
            ]
            self.assertTrue(all(ids), f"{slice_name} has an empty split")
            self.assertFalse(ids[0] & ids[1])
            self.assertFalse(ids[0] & ids[2])
            self.assertFalse(ids[1] & ids[2])
            total += sum(len(part) for part in ids)
        self.assertEqual(total, 48)

    def test_manifest_counts_match_and_reload_is_stable(self):
        bundle = materialize_datasets(self.spec, self.run_dir)
        for name, counts in bundle.counts.items():
            self.assertEqual(sum(counts.values()), 24)
            self.assertEqual(counts["report"], len(bundle.report_prompts[name]))

        again = load_bundle(self.spec, self.run_dir)
        self.assertEqual(again.counts, bundle.counts)
        self.assertEqual(
            {p.id for p in again.report_prompts["harmful_core"]},
            {p.id for p in bundle.report_prompts["harmful_core"]},
        )
        self.assertEqual(
            again.report_prompts["harmful_core"][0].text,
            bundle.report_prompts["harmful_core"][0].text,
        )

    def test_capability_items_load(self):
        items = load_capability_items(self.spec.evaluation.capability_tasks[0])
        self.assertEqual(len(items), 24)
        self.assertIn(items[0]["answer"], "ABCD")
        self.assertTrue(items[0]["choices"])

    def test_capability_jsonl_tolerates_utf8_bom(self):
        # PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM; a Windows
        # user's hand-written .jsonl should still load.
        path = self.tmp / "cap_bom.jsonl"
        body = json.dumps({"id": "q1", "question": "2+2?", "choices": ["3", "4"], "answer": "B"})
        path.write_text(body + "\n", encoding="utf-8-sig")
        task = self.spec.evaluation.capability_tasks[0]
        task.path_resolved = path
        items = load_capability_items(task)
        self.assertEqual(items[0]["id"], "q1")


if __name__ == "__main__":
    unittest.main()
