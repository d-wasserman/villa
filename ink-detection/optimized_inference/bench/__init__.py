"""
Benchmark harness for optimized_inference (2 um ink detection).

Importing this package installs the no-upload guard, so every process that
touches the harness (including spawned DataLoader workers, which unpickle
``bench.no_upload_guard.worker_init``) runs with it. See bench/README.md.
"""
import os
import sys
from pathlib import Path

# albumentations checks PyPI for a newer version on import (in every spawned
# DataLoader worker too). The guard would block it; skip the attempt entirely.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench import no_upload_guard  # noqa: E402

no_upload_guard.install()
