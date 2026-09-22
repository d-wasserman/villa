import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch

    import bench  # noqa: F401  (installs the guard)
    from bench import harness, schema, stats, workloads
except ImportError:  # CPU utility environments without torch
    harness = None


def _tiny(**overrides):
    cfg = dict(model="stub", in_chans=4, tile_size=16, stride=8, size=64, batch_size=2,
               workers=0, prefetch_factor=2, warmup=1, repeats=2, target_tag="test",
               sample_interval_ms=100)
    cfg.update(overrides)
    return harness.RunConfig(**cfg)


@unittest.skipIf(harness is None, "torch and inference dependencies not installed")
class HarnessTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def test_w1_stub_run_matches_schema(self):
        doc = harness.run(_tiny(workload="W1", mode="attribution"))
        self.assertEqual(schema.validate(doc), [])
        t = doc["timings"]
        self.assertEqual(t["tiles_total"], 49)  # 7 x 7 grid on 64 px at stride 8
        # Rows 0-15 are empty; the first batches are all-empty and skipped.
        self.assertTrue(all(n < 49 for n in t["tiles_forwarded"]))
        self.assertTrue(doc["quality"]["outputs_identical_across_repeats"])
        self.assertIsNotNone(t["stage_coverage"])
        self.assertEqual(doc["network"]["hosts_contacted"], [])
        self.assertTrue(doc["network"]["guard_active"])
        self.assertFalse(doc["config"]["autocast"])  # CPU default is fp32
        json.dumps(doc)

    def test_w0_stub_run_matches_schema(self):
        doc = harness.run(_tiny(workload="W0", w0_iters=2))
        self.assertEqual(schema.validate(doc), [])
        self.assertEqual(doc["timings"]["tiles_forwarded"], [4, 4])
        self.assertEqual(doc["quality"]["nonfinite_outputs"], 0)

    def test_reference_numerics_are_deterministic(self):
        doc = harness.run(_tiny(workload="W1", numerics="reference", compile_mode="default"))
        self.assertEqual(schema.validate(doc), [])
        self.assertTrue(doc["config"]["deterministic"])
        self.assertEqual(doc["config"]["compile_mode"], "off")
        self.assertTrue(doc["quality"]["outputs_identical_across_repeats"])

    def test_global_state_is_restored_after_run(self):
        import inference

        original = inference.create_inference_dataloader
        cfg_before = dict(vars(inference.CFG))
        benchmark_before = torch.backends.cudnn.benchmark
        harness.run(_tiny(workload="W1", numerics="reference", repeats=1, warmup=0))
        self.assertIs(inference.create_inference_dataloader, original)
        self.assertEqual(dict(vars(inference.CFG)), cfg_before)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(torch.backends.cudnn.benchmark, benchmark_before)


@unittest.skipIf(harness is None, "torch and inference dependencies not installed")
class SchemaAndStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = harness.run(_tiny(workload="W0", w0_iters=1, repeats=1, warmup=0))

    def test_schema_detects_missing_and_wrong_fields(self):
        doc = json.loads(json.dumps(self.doc))
        del doc["environment"]["torch_version"]
        doc["config"]["batch_size"] = "2"
        doc["config"]["mode"] = "fast"
        errors = schema.validate(doc)
        self.assertTrue(any("environment.torch_version" in e for e in errors))
        self.assertTrue(any("config.batch_size" in e for e in errors))
        doc = json.loads(json.dumps(self.doc))
        doc["config"]["mode"] = "fast"
        self.assertTrue(any("config.mode" in e for e in schema.validate(doc)))

    def test_bool_is_not_accepted_as_int(self):
        doc = json.loads(json.dumps(self.doc))
        doc["config"]["seed"] = True
        self.assertTrue(any("config.seed" in e for e in schema.validate(doc)))

    def test_compare_pairs(self):
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [0.90, 0.91, 0.89, 0.90, 0.90]
        r = stats.compare_pairs(a, b)
        self.assertAlmostEqual(r["ratio_b_over_a"], 0.9, places=2)
        self.assertTrue(r["significant"])
        lo, hi = r["t_interval"]
        self.assertLess(hi, 1.0)
        noisy = stats.compare_pairs(a, [0.8, 1.2, 0.9, 1.1, 1.0])
        self.assertFalse(noisy["significant"])

    def test_summarize(self):
        s = stats.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((s["n"], s["mean"], s["p50"], s["min"], s["max"]), (4, 2.5, 2.5, 1.0, 4.0))
        self.assertIsNone(stats.summarize([])["mean"])

    def test_w1_layers_bands(self):
        layers = workloads.make_w1_layers(64, 3, seed=0)
        self.assertFalse(layers[:16].any())
        self.assertFalse(layers[16:32, :32].any())
        self.assertTrue((layers[16:32, 32:] > 0).all())
        self.assertTrue((layers[32:] > 0).all())
        self.assertEqual(workloads.valid_pixel_count(layers), 16 * 32 + 32 * 64)


if __name__ == "__main__":
    unittest.main()
