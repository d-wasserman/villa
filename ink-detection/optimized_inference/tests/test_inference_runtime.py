import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    import torch.nn as nn

    import inference  # noqa: E402
except ImportError:  # CPU utility environments without torch
    inference = None


class _AutocastProbe(nn.Module):
    """Stub model that records whether CPU autocast was active during forward."""

    def __init__(self):
        super().__init__()
        self.autocast_seen = []

    def forward(self, x):
        self.autocast_seen.append(torch.is_autocast_enabled("cpu"))
        # (B, 1, C, H, W) -> (B, 1, H/4, W/4), like the ResNet3D decoder head
        return x.mean(dim=2)[:, :, ::4, ::4]


class _Wrapper:
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        return self.model(x)

    def get_output_scale_factor(self):
        return 4

    def eval(self):
        self.model.eval()


@unittest.skipIf(inference is None, "torch and inference dependencies not installed")
class InferenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(inference.CFG, k) for k in (
            "autocast", "in_chans", "size", "tile_size", "stride", "batch_size",
            "workers", "zarr_output_dir", "num_parts", "part_id",
        )}
        self._tmp = tempfile.TemporaryDirectory()
        cfg = inference.CFG
        cfg.in_chans, cfg.size, cfg.tile_size, cfg.stride = 4, 16, 16, 8
        cfg.batch_size, cfg.workers, cfg.num_parts, cfg.part_id = 2, 0, 1, 0
        cfg.zarr_output_dir = self._tmp.name

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(inference.CFG, k, v)
        self._tmp.cleanup()

    def _run(self):
        layers = np.random.default_rng(0).integers(1, 255, size=(32, 32, 4), dtype=np.uint8)
        probe = _AutocastProbe()
        inference.run_inference(layers, _Wrapper(probe), torch.device("cpu"))
        return probe.autocast_seen

    def test_autocast_defaults_on(self):
        self.assertTrue(inference.InferenceConfig.autocast)
        seen = self._run()
        self.assertTrue(seen)
        self.assertTrue(all(seen))

    def test_autocast_flag_disables_cpu_autocast(self):
        inference.CFG.autocast = False
        seen = self._run()
        self.assertTrue(seen)
        self.assertFalse(any(seen))


if __name__ == "__main__":
    unittest.main()
