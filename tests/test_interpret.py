"""Tests for the layer inspector's pure bookkeeping.

The model-loading path needs a real checkpoint and is exercised through the UI
smoke run; these tests pin the direction/offset arithmetic and path errors that
are easy to get subtly wrong.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tal.errors import SpecError
from tal.interpret import merge_layers, render_prompt, resolve_model_ref, summarize


def _row(index: int, projection: float, z: float) -> dict:
    return {
        "index": index,
        "label": "embed" if index == 0 else f"L{index}",
        "projection": projection,
        "z": z,
        "refusal_mass": 0.1 * index,
    }


class ResolveModelRefTests(unittest.TestCase):
    def test_hf_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_model_ref({"path": str(root)}), str(root.resolve()))

    def test_gguf_uses_eval_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf = root / "hf"
            hf.mkdir()
            (hf / "config.json").write_text("{}", encoding="utf-8")
            gguf = root / "model.gguf"
            gguf.write_bytes(b"GGUF")
            self.assertEqual(
                resolve_model_ref({"path": str(gguf), "eval_path": str(hf)}),
                str(hf.resolve()),
            )

    def test_gguf_without_eval_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            gguf.write_bytes(b"GGUF")
            with self.assertRaisesRegex(SpecError, "评测用 HF 目录"):
                resolve_model_ref({"path": str(gguf), "eval_path": ""})

    def test_directory_without_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SpecError, "config.json"):
                resolve_model_ref({"path": tmp})


class PromptRenderingTests(unittest.TestCase):
    def test_chat_template_is_preferred(self):
        class Tokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                system = messages[0]["content"] if len(messages) > 1 else ""
                return f"{system}|{messages[-1]['content']}"

        self.assertEqual(
            render_prompt(Tokenizer(), "hello", "/no_think"),
            "/no_think|hello",
        )

    def test_plain_fallback(self):
        class Tokenizer:
            chat_template = None

        self.assertEqual(render_prompt(Tokenizer(), "hello", "sys"), "sys\n\nhello")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.base = [
            _row(0, 0.1, 0.0),
            _row(1, 1.0, 0.5),
            _row(2, 2.0, 1.5),
            _row(3, 3.0, 3.0),
        ]

    def test_onset_peak_and_ablation(self):
        edited = [
            _row(0, 0.1, 0.0),
            _row(1, 0.9, 0.5),
            _row(2, -1.0, 1.5),
            _row(3, 0.5, 3.0),
        ]
        summary = summarize(self.base, edited, [{"layer": 1, "delta": -0.1}, {"layer": 2, "delta": -0.5}])
        self.assertEqual(summary["onset_layer"], 2)
        self.assertEqual(summary["peak_layer"], 3)
        self.assertEqual(summary["ablation_layer"], 2)
        self.assertAlmostEqual(summary["ablation_drop"], -3.0)
        self.assertAlmostEqual(summary["final_delta"], -2.5)
        self.assertEqual(summary["sweep_layer"], 2)

    def test_base_only_has_no_ablation_layer(self):
        summary = summarize(self.base, [])
        self.assertIsNone(summary["ablation_layer"])
        self.assertIsNone(summary["final_delta"])
        self.assertEqual(summary["n_layers"], 4)

    def test_merge_layers_aligns_by_index(self):
        edited = [_row(0, 0.2, 0.0), _row(1, 0.4, 0.4)]
        merged = merge_layers(self.base[:2], edited)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[1]["delta_projection"], -0.6)
        self.assertAlmostEqual(merged[1]["delta_refusal_mass"], -0.0)


if __name__ == "__main__":
    unittest.main()
