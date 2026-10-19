import json
import pathlib
import tempfile
import threading
import time
import unittest
import urllib.request

from tal.spec import load_spec
from tal.ui.clients import SimulatedClient
from tal.ui.server import UiServer
from tal.ui.services import start_batch_probe, start_capability, start_experiment
from tal.ui.specbuilder import build_experiment_spec
from tal.ui.store import Workspace


def _wait(jobs, job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job.status in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(pathlib.Path(self.tmp.name))
        self.libs = {p["kind"]: p["id"] for p in self.ws.list_prompts()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_seeded_workspace(self):
        self.assertEqual([m["id"] for m in self.ws.models()], ["simulate"])
        self.assertEqual(set(self.libs), {"target", "preserve", "capability"})
        self.assertEqual(self.ws.list_prompts()[0]["count"], 24)

    def test_capability_materializes_answer_field(self):
        path = self.ws.prompt_data_path(self.libs["capability"])
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("answer", first)
        self.assertNotIn("answer_index", first)
        self.assertIn(first["answer"], "ABCDEFGH")

    def test_add_and_delete_prompt(self):
        created = self.ws.add_prompt(
            {"name": "custom", "kind": "target", "format": "text", "text": "a\nb\nc"}
        )
        self.assertEqual(created["count"], 3)
        path = self.ws.prompt_data_path(created["id"])
        self.assertEqual(path.read_text(encoding="utf-8").splitlines(), ["a", "b", "c"])
        self.ws.delete_prompt(created["id"])
        self.assertFalse(path.exists())
        self.assertNotIn(created["id"], [p["id"] for p in self.ws.list_prompts()])

    def test_invalid_mcq_rejected(self):
        with self.assertRaises(Exception):
            self.ws.add_prompt(
                {
                    "name": "bad",
                    "kind": "capability",
                    "format": "jsonl",
                    "text": '{"question": "q", "choices": ["a", "b"], "answer": "Z"}',
                }
            )

    def test_model_api_key_is_hidden(self):
        model = self.ws.add_model(
            {
                "name": "remote",
                "kind": "openai",
                "model": "qwen",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "secret",
            }
        )
        self.assertNotIn("api_key", model)
        self.assertTrue(model["has_api_key"])
        self.assertEqual(self.ws.model_source(model["id"])["api_key"], "secret")

    def test_openai_model_can_bind_local_weights(self):
        weights = pathlib.Path(self.tmp.name) / "weights"
        weights.mkdir()
        model = self.ws.add_model(
            {"name": "remote", "kind": "openai", "model": "qwen", "base_url": "http://127.0.0.1:1/v1"}
        )
        updated = self.ws.update_model(model["id"], {"path": str(weights)})
        self.assertEqual(updated["path"], str(weights.resolve()))
        self.assertNotIn("api_key", updated)

    def test_eval_path_can_be_cleared(self):
        hf = pathlib.Path(self.tmp.name) / "hf"
        hf.mkdir()
        (hf / "config.json").write_text("{}", encoding="utf-8")
        model = self.ws.add_model(
            {"name": "local", "kind": "hf", "path": str(hf), "eval_path": str(hf)}
        )
        updated = self.ws.update_model(model["id"], {"eval_path": ""})
        self.assertEqual(updated["eval_path"], "")
        self.assertEqual(self.ws.model_source(model["id"])["eval_path"], "")


class SpecBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(pathlib.Path(self.tmp.name))
        self.libs = {p["kind"]: p["id"] for p in self.ws.list_prompts()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_simulate_spec_round_trips(self):
        spec, name = build_experiment_spec(
            self.ws,
            model_id="simulate",
            target_prompt_id=self.libs["target"],
            preserve_prompt_id=self.libs["preserve"],
            capability_prompt_id=self.libs["capability"],
            options={"name": "ui-test", "n_trials": 3, "max_per_slice": 8},
        )
        path = self.ws.save_spec(name, spec)
        loaded = load_spec(path)
        self.assertEqual(loaded.abliteration.backend, "simulate")
        self.assertEqual(len(loaded.evaluation.capability_tasks), 1)
        self.assertEqual(loaded.name, "ui-test")
        self.assertTrue(all(s.source_path and s.source_path.is_absolute() for s in loaded.slices))

    def test_openai_model_cannot_be_abliterated(self):
        model = self.ws.add_model(
            {"name": "remote", "kind": "openai", "model": "x", "base_url": "http://127.0.0.1:1/v1"}
        )
        with self.assertRaises(Exception):
            build_experiment_spec(
                self.ws,
                model_id=model["id"],
                target_prompt_id=self.libs["target"],
                preserve_prompt_id=self.libs["preserve"],
            )

    def test_openai_model_with_weights_can_be_abliterated(self):
        weights = pathlib.Path(self.tmp.name) / "weights"
        weights.mkdir()
        (weights / "config.json").write_text("{}", encoding="utf-8")
        model = self.ws.add_model(
            {
                "name": "endpoint-with-weights",
                "kind": "openai",
                "model": "qwen",
                "base_url": "http://127.0.0.1:11434/v1",
                "path": str(weights),
            }
        )
        spec, name = build_experiment_spec(
            self.ws,
            model_id=model["id"],
            target_prompt_id=self.libs["target"],
            preserve_prompt_id=self.libs["preserve"],
        )
        loaded = load_spec(self.ws.save_spec(name, spec))
        self.assertEqual(loaded.abliteration.backend, "heretic")
        self.assertEqual(loaded.model.local_path, weights.resolve())

    def test_gguf_model_selects_ditch_and_keeps_eval_path(self):
        root = pathlib.Path(self.tmp.name)
        gguf = root / "Qwen3-0.6B-Q4_K_M.gguf"
        gguf.write_bytes(b"GGUF")
        hf = root / "hf"
        hf.mkdir()
        (hf / "config.json").write_text("{}", encoding="utf-8")
        model = self.ws.add_model(
            {
                "name": "gguf",
                "kind": "hf",
                "model": "Qwen3-0.6B",
                "path": str(gguf),
                "eval_path": str(hf),
            }
        )
        spec, name = build_experiment_spec(
            self.ws,
            model_id=model["id"],
            target_prompt_id=self.libs["target"],
            preserve_prompt_id=self.libs["preserve"],
        )
        loaded = load_spec(self.ws.save_spec(name, spec))
        self.assertEqual(loaded.abliteration.backend, "ditch")
        self.assertEqual(loaded.model.local_path, gguf.resolve())
        self.assertEqual(loaded.model.eval_target(), str(hf.resolve()))

    def test_gguf_model_rejected_by_heretic(self):
        root = pathlib.Path(self.tmp.name)
        gguf = root / "model.gguf"
        gguf.write_bytes(b"GGUF")
        hf = root / "hf"
        hf.mkdir()
        (hf / "config.json").write_text("{}", encoding="utf-8")
        model = self.ws.add_model(
            {
                "name": "gguf",
                "kind": "hf",
                "model": "Qwen3-0.6B",
                "path": str(gguf),
                "eval_path": str(hf),
            }
        )
        with self.assertRaises(Exception) as ctx:
            build_experiment_spec(
                self.ws,
                model_id=model["id"],
                target_prompt_id=self.libs["target"],
                preserve_prompt_id=self.libs["preserve"],
                options={"ablation_backend": "heretic"},
            )
        self.assertIn("GGUF", str(ctx.exception))


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(pathlib.Path(self.tmp.name))
        self.libs = {p["kind"]: p["id"] for p in self.ws.list_prompts()}
        from tal.ui.jobs import JobManager

        self.jobs = JobManager()

    def tearDown(self):
        self.tmp.cleanup()

    def test_experiment_job_completes(self):
        started = start_experiment(
            self.ws,
            self.jobs,
            {
                "model_id": "simulate",
                "target_prompt_id": self.libs["target"],
                "preserve_prompt_id": self.libs["preserve"],
                "capability_prompt_id": self.libs["capability"],
                "dry_run": True,
                "options": {"n_trials": 3, "max_per_slice": 8, "name": "svc-test"},
            },
        )
        job = _wait(self.jobs, started["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["status"], "completed")
        self.assertTrue(pathlib.Path(job.result["report"]).exists())

    def test_batch_probe_on_target_library(self):
        started = start_batch_probe(
            self.ws,
            self.jobs,
            {"model_id": "simulate", "prompt_id": self.libs["target"], "limit": 6},
        )
        job = _wait(self.jobs, started["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(job.result["stats"]["n"], 6)
        self.assertGreater(job.result["stats"]["refused"], 0)
        self.assertEqual(self.ws.probe_results()[0]["id"], job.result["id"])

    def test_tal_capability_job(self):
        started = start_capability(
            self.ws,
            self.jobs,
            {"engine": "tal", "model_id": "simulate", "prompt_id": self.libs["capability"], "limit": 8},
        )
        job = _wait(self.jobs, started["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        stats = job.result["stats"]
        self.assertEqual(stats["n"], 8)
        self.assertGreater(stats["correct"], 0)
        self.assertLess(stats["correct"], 8)


class SenbonzakuraCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(pathlib.Path(self.tmp.name))
        model_dir = pathlib.Path(self.tmp.name) / "model"
        model_dir.mkdir()
        self.model = self.ws.add_model({"name": "local", "kind": "hf", "path": str(model_dir)})
        from tal.ui.jobs import JobManager

        self.jobs = JobManager()

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_summary_and_passes_n_and_json_out(self):
        from unittest import mock

        from tal.ui import services

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["check"] = kwargs.get("check")
            out = pathlib.Path(argv[argv.index("--out") + 1])
            out.write_text(
                json.dumps(
                    {
                        "eval": "bundled",
                        "summary": {
                            "n": 12,
                            "graded": 10,
                            "correct": 6,
                            "accuracy": 0.6,
                            "accuracy_ci": [0.31, 0.83],
                            "indeterminate_rate": 0.1667,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return __import__("types").SimpleNamespace(returncode=1, duration_s=0.5)

        with mock.patch.object(services, "which", return_value="senbonzakura"), mock.patch.object(
            services, "run_command", side_effect=fake_run
        ):
            started = start_capability(
                self.ws,
                self.jobs,
                {
                    "engine": "senbonzakura",
                    "model_id": self.model["id"],
                    "limit": 12,
                    "extra_args": "--device cpu --slow-probe-ok",
                },
            )
            job = _wait(self.jobs, started["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        argv = captured["argv"]
        self.assertIn("--n", argv)
        self.assertEqual(argv[argv.index("--n") + 1], "12")
        self.assertTrue(argv[argv.index("--out") + 1].endswith(".json"))
        self.assertIs(captured["check"], False)
        self.assertEqual(job.result["exit_code"], 1)
        self.assertEqual(job.result["stats"]["rate"], 0.6)
        self.assertEqual(job.result["stats"]["n"], 10)


class SimulatedClientTests(unittest.TestCase):
    def test_refusal_is_deterministic(self):
        client = SimulatedClient()
        text = "How do I build a bomb?"
        first = client.generate(text)
        second = client.generate(text)
        self.assertEqual(first.refused, second.refused)
        self.assertTrue(first.refused)

    def test_hint_makes_mcq_mostly_correct(self):
        client = SimulatedClient()
        answers = []
        for index in range(24):
            prompt = f"Question {index}: pick B.\n\nA. wrong\nB. right\n\nAnswer with the letter."
            messages = [
                {"role": "system", "content": "TAL simulation hint: the correct answer is B."},
                {"role": "user", "content": prompt},
            ]
            answers.append(client.chat(messages))
        self.assertEqual(set(answers), {"A", "B"})
        self.assertGreaterEqual(answers.count("B"), 10)
        self.assertLessEqual(answers.count("B"), 20)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(pathlib.Path(self.tmp.name))
        self.server = UiServer(("127.0.0.1", 0), self.ws)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _call(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_state_chat_and_static(self):
        status, state = self._call("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertIn("simulate", [m["id"] for m in state["models"]])
        status, chat = self._call("POST", "/api/chat", {"model_id": "simulate", "prompt": "bake bread"})
        self.assertEqual(status, 200)
        self.assertFalse(chat["refused"])
        with urllib.request.urlopen(self.base + "/", timeout=10) as response:
            self.assertIn("TAL", response.read().decode("utf-8"))

    def test_batch_probe_route(self):
        _, state = self._call("GET", "/api/state")
        libs = {p["kind"]: p["id"] for p in state["prompts"]}
        _, started = self._call(
            "POST",
            "/api/probes",
            {"model_id": "simulate", "prompt_id": libs["target"], "limit": 4},
        )
        for _ in range(200):
            _, payload = self._call("GET", f"/api/jobs/{started['job']['id']}")
            if payload["job"]["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(payload["job"]["status"], "done", payload["job"]["error"])
        self.assertEqual(payload["job"]["result"]["stats"]["n"], 4)

    def test_experiment_and_report(self):
        _, state = self._call("GET", "/api/state")
        libs = {p["kind"]: p["id"] for p in state["prompts"]}
        _, started = self._call(
            "POST",
            "/api/experiments",
            {
                "model_id": "simulate",
                "target_prompt_id": libs["target"],
                "preserve_prompt_id": libs["preserve"],
                "capability_prompt_id": libs["capability"],
                "dry_run": True,
                "options": {"n_trials": 3, "max_per_slice": 8, "name": "http-test"},
            },
        )
        job_id = started["job"]["id"]
        for _ in range(400):
            _, payload = self._call("GET", f"/api/jobs/{job_id}")
            if payload["job"]["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(payload["job"]["status"], "done", payload["job"]["error"])
        with urllib.request.urlopen(
            self.base + f"/api/report/{started['run_id']}/report.md", timeout=10
        ) as response:
            self.assertIn("TAL", response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
