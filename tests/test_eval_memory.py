"""Preflight checks that keep a large checkpoint from killing the UI process."""

from __future__ import annotations

import unittest

from tal.eval.hf import _preflight_error

GB = 1024**3


class PreflightTests(unittest.TestCase):
    def test_large_bf16_model_is_rejected_on_small_gpu(self):
        problem = _preflight_error(
            8 * GB,
            free_vram=int(7.5 * GB),
            available_ram=int(1.5 * GB),
            device="cuda",
            quantized=False,
        )
        self.assertIsNotNone(problem)
        self.assertIn("4-bit", problem)

    def test_small_model_still_loads(self):
        self.assertIsNone(
            _preflight_error(
                1 * GB,
                free_vram=int(7.5 * GB),
                available_ram=int(1.5 * GB),
                device="cuda",
                quantized=False,
            )
        )

    def test_quantized_load_still_needs_host_memory(self):
        problem = _preflight_error(
            8 * GB,
            free_vram=int(7.5 * GB),
            available_ram=int(5 * GB),
            device="cuda",
            quantized=True,
        )
        self.assertIsNotNone(problem)
        self.assertIn("流经内存", problem)

    def test_cpu_counts_float32_memory(self):
        self.assertIsNotNone(
            _preflight_error(
                4 * GB,
                free_vram=0,
                available_ram=int(6 * GB),
                device="cpu",
                quantized=False,
            )
        )

    def test_cpu_auto_dtype_can_use_a_smaller_factor(self):
        self.assertIsNone(
            _preflight_error(
                3 * GB,
                free_vram=0,
                available_ram=int(6 * GB),
                device="cpu",
                quantized=False,
                cpu_factor=1.3,
            )
        )

    def test_unknown_sizes_and_memory_skip_the_check(self):
        self.assertIsNone(
            _preflight_error(
                0,
                free_vram=0,
                available_ram=0,
                device="cuda",
                quantized=False,
            )
        )
        self.assertIsNone(
            _preflight_error(
                8 * GB,
                free_vram=0,
                available_ram=0,
                device="cuda",
                quantized=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
