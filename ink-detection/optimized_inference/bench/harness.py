"""
Benchmark harness core: builds CFG, the profiler and the model, then calls
the production code (inference.configure_runtime, inference.run_inference)
directly. It never calls entrypoint.py step functions and never creates an
AWS client (see bench/no_upload_guard.py).

Timing modes (never mixed in one result file):
  throughput   production-like, no extra CUDA syncs. Feeds the headline metric.
  attribution  per-stage timing with CUDA syncs (profiler level "detailed")
               plus DataLoader wait time. Slower; answers where time goes.

Numerics:
  production   autocast (fp16) on CUDA; fp32 on CPU unless --autocast on.
  reference    fp32, torch.use_deterministic_algorithms(True), cudnn.benchmark
               off, no compile. For proving plumbing changes are exact.
"""
from __future__ import annotations

import gc
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from bench import no_upload_guard
from bench import schema
from bench.stats import cm2, summarize

ROOT = Path(__file__).resolve().parents[1]
# Main-thread intervals that should add up to the run_inference wall time.
# setup: run start to loop start (dataset, loader, full-size accumulators).
# loader_start: creating the loader iterator (spawns DataLoader workers).
# teardown: loop end to return, minus zarr_write (loader and worker shutdown, gc).
PROFILER_STAGES = ("host_to_device_seconds", "forward_seconds", "device_to_host_seconds",
                   "postprocess_seconds", "zarr_write_seconds")
MAIN_STAGES = ("setup_seconds", "loader_start_seconds", "loader_wait_seconds") + PROFILER_STAGES + ("teardown_seconds",)
STAGE_TOLERANCE = 0.10


@dataclass
class RunConfig:
    workload: str = "W1"
    target_tag: str = "local"
    mode: str = "throughput"
    numerics: str = "production"
    autocast: str = "auto"
    device: str = "auto"
    model: str = "resnet3d-152-3d-decoder"
    weights: str = "random"
    batch_size: int = 2
    workers: int = 2
    prefetch_factor: int = 2
    tile_size: int = 256
    stride: int = 128
    in_chans: int = 62
    size: int = 1024
    w0_iters: int = 3
    compile_mode: str = "off"
    inductor_cache: str = "cold"
    inductor_cache_dir: str = ""
    warmup: int = 1
    repeats: int = 5
    seed: int = 0
    pixel_um: float = 2.4
    out_dir: str = ""
    allow_hosts: List[str] = field(default_factory=list)
    torch_trace: bool = False
    sample_interval_ms: int = 500


# ----------------------------------------------------------------------------- environment
def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _cpu_model() -> Optional[str]:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or None
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def collect_environment(device) -> Dict[str, Any]:
    import torch

    try:
        import psutil

        ram = int(psutil.virtual_memory().total)
    except Exception:
        ram = None
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    gpu_name = capability = driver = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        capability = "%d.%d" % torch.cuda.get_device_capability(device)
        try:
            import pynvml

            pynvml.nvmlInit()
            driver = pynvml.nvmlSystemGetDriverVersion()
            driver = driver.decode() if isinstance(driver, bytes) else str(driver)
        except Exception:
            driver = None
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "device": device.type,
        "gpu_name": gpu_name,
        "gpu_compute_capability": capability,
        "gpu_driver": driver,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": int(cudnn) if cudnn is not None else None,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "ram_bytes": ram,
        "cpu_count_visible": int(cpus),
    }


# ----------------------------------------------------------------------------- model
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(cfg: RunConfig, device):
    """Returns (wrapper, weights_sha256)."""
    import torch
    import model_resnet3d_3d_decoder as m3d

    from bench.workloads import StubModel

    if cfg.model == "stub":
        torch.manual_seed(cfg.seed)
        net = StubModel().to(device).eval()
        return m3d.ResNet3DDecoderWrapper(net, device), None
    if cfg.model != "resnet3d-152-3d-decoder":
        raise ValueError(f"unsupported model {cfg.model!r}")
    if cfg.weights == "random":
        torch.manual_seed(cfg.seed)
        net = m3d.RegressionModel(with_norm=False).to(device).eval()
        return m3d.ResNet3DDecoderWrapper(net, device), None

    # Real weights: production load_model uses strict=False and only logs
    # missing keys, so a partial load gives plausible garbage. Fail instead.
    checkpoint = torch.load(cfg.weights, map_location="cpu", weights_only=False, mmap=True)
    ckpt_keys = set(m3d._strip_known_prefixes(m3d._extract_state_dict(checkpoint)).keys())
    del checkpoint
    wrapper = m3d.load_model(cfg.weights, device, num_frames=cfg.in_chans)
    net = wrapper.model.module if isinstance(wrapper.model, torch.nn.DataParallel) else wrapper.model
    model_keys = set(net.state_dict().keys())
    missing, unexpected = sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint key mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    return wrapper, _sha256_file(cfg.weights)


# ----------------------------------------------------------------------------- loader instrumentation
class _LoaderTimes:
    def __init__(self):
        self.iter_start: Optional[float] = None
        self.iter_ready: Optional[float] = None
        self.loop_end: Optional[float] = None
        # per batch: (request_time, yield_time, n_tiles, forwarded)
        self.batches: List[tuple] = []

    @property
    def wait_seconds(self) -> float:
        return sum(y - r for r, y, _, _ in self.batches)

    def first_batch_seconds(self) -> Optional[float]:
        """Loop start to the end of the first batch's processing (includes worker start and recompiles)."""
        if not self.batches or self.iter_start is None:
            return None
        end = self.batches[1][0] if len(self.batches) > 1 else self.loop_end
        return None if end is None else end - self.iter_start

    def steady_state_tiles_per_second(self) -> Optional[float]:
        """Forwarded tiles per second after the first batch."""
        if len(self.batches) < 2 or self.loop_end is None:
            return None
        tiles = sum(n for _, _, n, fwd in self.batches[1:] if fwd)
        window = self.loop_end - self.batches[1][0]
        return tiles / window if window > 0 else None


class _TimedLoader:
    """Wraps a DataLoader, timing how long the main loop waits for each batch."""

    def __init__(self, loader, times: _LoaderTimes):
        self._loader = loader
        self._times = times
        self.dataset = loader.dataset

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        times = self._times
        times.iter_start = time.perf_counter()
        it = iter(self._loader)
        times.iter_ready = time.perf_counter()
        while True:
            requested = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                times.loop_end = requested
                return
            images, _, valids = batch
            b = int(images.size(0))
            # Mirrors predict_fn's whole-batch skip rule, only to count forwarded tiles.
            times.batches.append((requested, time.perf_counter(), b, bool(valids.view(b, -1).any())))
            yield batch


@contextmanager
def _instrumented_dataloader(times: _LoaderTimes, workers: int) -> Iterator[None]:
    """
    Temporarily wraps inference.create_inference_dataloader so the returned
    loader (a) installs the no-upload guard in spawned workers and (b) records
    loader wait times. Restores the original on exit.
    """
    import inference

    original = inference.create_inference_dataloader

    def factory(*args, **kwargs):
        loader, pred_shape, info = original(*args, **kwargs)
        if workers > 0:
            loader.worker_init_fn = no_upload_guard.worker_init
        return _TimedLoader(loader, times), pred_shape, info

    inference.create_inference_dataloader = factory
    try:
        yield
    finally:
        inference.create_inference_dataloader = original


# ----------------------------------------------------------------------------- profiler
def _make_profiler(cfg: RunConfig, local_root: Path, part_id: int = 0):
    from profiling import WorkflowProfiler

    class BenchProfiler(WorkflowProfiler):
        # Detailed level also starts torch.profiler for the first 20 batches,
        # which distorts attribution timings. Only allow it on request.
        def enable_torch_profiler(self) -> bool:
            return cfg.torch_trace and super().enable_torch_profiler()

    return BenchProfiler(
        level="detailed" if cfg.mode == "attribution" else "basic",
        sample_interval_ms=cfg.sample_interval_ms,
        raw_root=None,
        local_root=str(local_root),
        step_name="inference",
        template_name="bench",
        part_id=part_id,
        metadata={"bench": True},
        runtime_parameters={},
    )


def _telemetry(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gpu_temperature_c_max": metrics.get("gpu_temperature_celsius_max"),
        "gpu_sm_clock_mhz_min": metrics.get("gpu_sm_clock_mhz_min"),
        "gpu_utilization_percent_avg": metrics.get("gpu_utilization_percent_avg"),
        "peak_vram_bytes": metrics.get("torch_cuda_max_memory_allocated_bytes"),
        "peak_host_rss_bytes": metrics.get("process_rss_bytes_peak"),
    }


# ----------------------------------------------------------------------------- setup
def _resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _setup_inductor_cache(cfg: RunConfig, run_tmp: Path) -> str:
    """Point Inductor and Triton at the requested cache; returns cache state."""
    if cfg.compile_mode == "off":
        return "n/a"
    if cfg.inductor_cache == "cold":
        cache = run_tmp / "inductor_cache"
    else:
        cache = Path(cfg.inductor_cache_dir or Path.home() / ".cache" / "villa-bench" / "inductor").expanduser()
    state = "populated" if cache.is_dir() and any(cache.iterdir()) else "empty"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    return state


def _reset_cfg(cfg: RunConfig, autocast: bool) -> Dict[str, Any]:
    import inference

    c = inference.CFG
    vars(c).clear()  # drop previous run's overrides; class defaults remain
    c.model_type = cfg.model
    c.in_chans = cfg.in_chans
    c.tile_size = cfg.tile_size
    c.size = cfg.tile_size
    c.stride = cfg.stride
    c.batch_size = cfg.batch_size
    c.workers = cfg.workers
    c.prefetch_factor = cfg.prefetch_factor
    c.num_parts = 1
    c.part_id = 0
    c.autocast = autocast
    return {k: getattr(c, k) for k in dir(type(c)) if not k.startswith("_")}


def _hash_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- workloads
def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_w0_repeat(cfg: RunConfig, model, device, autocast: bool, x) -> Dict[str, Any]:
    import torch

    amp = "cuda" if device.type == "cuda" else "cpu"
    first = None
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(cfg.w0_iters):
            with torch.autocast(device_type=amp, enabled=autocast):
                y = model.forward(x)
            if i == 0:
                _sync(device)
                first = time.perf_counter() - t0
    y = y.float()
    _sync(device)
    wall = time.perf_counter() - t0
    out = y.cpu().numpy()
    import numpy as np

    return {
        "wall": wall,
        "tiles_forwarded": cfg.w0_iters * cfg.batch_size,
        "first_batch_seconds": first,
        "steady_tps": None,
        "stages": None,
        "hash": _hash_arrays(out),
        "nonfinite": int(np.count_nonzero(~np.isfinite(out))),
    }


def _run_w1_repeat(cfg: RunConfig, model, device, layers, rep_dir: Path, profiler) -> Dict[str, Any]:
    import inference
    import zarr

    inference.CFG.zarr_output_dir = str(rep_dir / "partitions")
    times = _LoaderTimes()
    with _instrumented_dataloader(times, cfg.workers):
        t0 = time.perf_counter()
        result = inference.run_inference(layers, model, device, profiler=profiler)
        _sync(device)
        wall = time.perf_counter() - t0
    gc.collect()
    pred = zarr.open(result["mask_pred"], mode="r")[:]
    count = zarr.open(result["mask_count"], mode="r")[:]
    t_end = t0 + wall
    return {
        "wall": wall,
        "intervals": {
            "setup_seconds": times.iter_start - t0,
            "loader_start_seconds": times.iter_ready - times.iter_start,
            "after_loop_seconds": t_end - times.loop_end,
        },
        "tiles_forwarded": int(result.get("partition_tiles") or 0),
        "first_batch_seconds": times.first_batch_seconds(),
        "steady_tps": times.steady_state_tiles_per_second(),
        "loader_wait_seconds": times.wait_seconds,
        "hash": _hash_arrays(pred, count),
        "nonfinite": None,
    }


def run(cfg: RunConfig) -> Dict[str, Any]:
    if cfg.workload not in ("W0", "W1"):
        raise ValueError(f"unknown workload {cfg.workload!r}")
    if cfg.mode not in ("throughput", "attribution"):
        raise ValueError(f"unknown mode {cfg.mode!r}")
    if cfg.numerics not in ("production", "reference"):
        raise ValueError(f"unknown numerics {cfg.numerics!r}")
    no_upload_guard.install(cfg.allow_hosts)
    no_upload_guard.reset_attempts()
    # Record the code state at start, before a long run can overlap new commits.
    git_commit = _git("rev-parse", "HEAD")
    git_dirty = (_git("status", "--porcelain", "--untracked-files=no", "--", ".") or "") != "" if git_commit else None

    deterministic = cfg.numerics == "reference"
    if deterministic:
        # Must be set before CUDA initializes for deterministic cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cfg.compile_mode = "off"

    import torch
    import inference  # sets cudnn.benchmark = True at import, like production

    device = _resolve_device(cfg.device)
    saved_torch = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    saved_cfg = dict(vars(inference.CFG))
    run_tmp = Path(tempfile.mkdtemp(prefix="villa-bench-"))
    try:
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            autocast = False
        elif cfg.autocast == "auto":
            autocast = device.type == "cuda"
        else:
            autocast = cfg.autocast == "on"
        cache_state = _setup_inductor_cache(cfg, run_tmp)
        cfg_values = _reset_cfg(cfg, autocast)

        t0 = time.perf_counter()
        model, weights_sha = build_model(cfg, device)
        _sync(device)
        model_build = time.perf_counter() - t0

        compile_seconds = None
        if cfg.compile_mode != "off":
            t0 = time.perf_counter()
            inference.configure_runtime(model, device, compile_enabled=True, compile_mode=cfg.compile_mode)
            _sync(device)
            compile_seconds = time.perf_counter() - t0
        else:
            inference.configure_runtime(model, device, compile_enabled=False)

        from bench import workloads

        if cfg.workload == "W0":
            x = workloads.make_w0_input(cfg.batch_size, cfg.in_chans, cfg.tile_size, cfg.seed).to(device)
            input_shape = list(x.shape)
            # One tile contributes stride^2 new pixels to a fully covered segment.
            gross_px = net_px = cfg.w0_iters * cfg.batch_size * cfg.stride ** 2
            tiles_total = cfg.w0_iters * cfg.batch_size
        else:
            layers = workloads.make_w1_layers(cfg.size, cfg.in_chans, cfg.seed)
            input_shape = list(layers.shape)
            gross_px = cfg.size * cfg.size
            net_px = workloads.valid_pixel_count(layers)
            tiles_total = len(inference._grid_1d(cfg.size, cfg.tile_size, cfg.stride)) ** 2

        reps: List[Dict[str, Any]] = []
        first_batch = None
        for i in range(cfg.warmup + cfg.repeats):
            rep_dir = run_tmp / f"rep{i:02d}"
            rep_dir.mkdir()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            profiler = _make_profiler(cfg, rep_dir / "profiling")
            status = "failed"
            try:
                if cfg.workload == "W0":
                    rec = _run_w0_repeat(cfg, model, device, autocast, x)
                else:
                    rec = _run_w1_repeat(cfg, model, device, layers, rep_dir, profiler)
                status = "succeeded"
            finally:
                profiler.flush(status)
            metrics = dict(profiler.metrics)
            if i == 0:
                first_batch = rec["first_batch_seconds"]
            if cfg.mode == "attribution" and cfg.workload == "W1":
                stages = {k: float(metrics.get(k) or 0.0) for k in PROFILER_STAGES}
                stages["setup_seconds"] = rec["intervals"]["setup_seconds"]
                stages["loader_start_seconds"] = rec["intervals"]["loader_start_seconds"]
                stages["loader_wait_seconds"] = rec["loader_wait_seconds"]
                stages["teardown_seconds"] = rec["intervals"]["after_loop_seconds"] - stages["zarr_write_seconds"]
                rec["stages"] = stages
                rec["worker_preprocess_seconds"] = metrics.get("preprocess_seconds")
                rec["worker_summaries_merged"] = len(list((rep_dir / "profiling" / "workers").glob("worker-summary-*.json")))
            rec["telemetry"] = _telemetry(metrics)
            if i >= cfg.warmup:
                reps.append(rec)
            shutil.rmtree(rep_dir / "partitions", ignore_errors=True)

        walls = [r["wall"] for r in reps]
        tiles = [r["tiles_forwarded"] for r in reps]
        stage_seconds = coverage = accepted = None
        if cfg.mode == "attribution" and cfg.workload == "W1":
            stage_seconds = {k: summarize([r["stages"][k] for r in reps])["mean"] for k in MAIN_STAGES}
            stage_seconds["worker_preprocess_seconds_total"] = summarize(
                [r["worker_preprocess_seconds"] for r in reps if r.get("worker_preprocess_seconds") is not None]
            )["mean"]
            stage_seconds["worker_summaries_merged"] = summarize([r["worker_summaries_merged"] for r in reps])["mean"]
            coverage = sum(stage_seconds[k] for k in MAIN_STAGES) / summarize(walls)["mean"]
            accepted = abs(1.0 - coverage) <= STAGE_TOLERANCE
        steady = [r["steady_tps"] for r in reps if r["steady_tps"] is not None]
        per_rep_tel = [dict(r["telemetry"], wall_seconds=r["wall"]) for r in reps]

        def _agg(key, fn):
            vals = [t[key] for t in per_rep_tel if t[key] is not None]
            return fn(vals) if vals else None

        hashes = [r["hash"] for r in reps]
        nonfinite = [r["nonfinite"] for r in reps if r["nonfinite"] is not None]
        doc = {
            "identity": {
                "schema_version": schema.SCHEMA_VERSION,
                "run_id": uuid.uuid4().hex,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "target_tag": cfg.target_tag,
            },
            "environment": collect_environment(device),
            "config": {
                "workload": cfg.workload,
                "mode": cfg.mode,
                "numerics": cfg.numerics,
                "autocast": autocast,
                "deterministic": deterministic,
                "compile_mode": cfg.compile_mode,
                "inductor_cache": cfg.inductor_cache if cfg.compile_mode != "off" else "n/a",
                "inductor_cache_state": cache_state,
                "model": cfg.model,
                "weights": "random" if cfg.weights == "random" else "real",
                "weights_sha256": weights_sha,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "workers": cfg.workers,
                "prefetch_factor": cfg.prefetch_factor,
                "tile_size": cfg.tile_size,
                "stride": cfg.stride,
                "in_chans": cfg.in_chans,
                "input_shape": input_shape,
                "pixel_um": cfg.pixel_um,
                "warmup": cfg.warmup,
                "repeats": cfg.repeats,
                "crop_ids": [],
                "cfg": _jsonable(cfg_values),
                "cli": _jsonable(asdict(cfg)),
            },
            "timings": {
                "model_build_seconds": model_build,
                "compile_seconds": compile_seconds,
                "first_batch_seconds": first_batch,
                "repeat_wall_seconds": walls,
                "tiles_total": tiles_total,
                "tiles_forwarded": tiles,
                "tiles_per_second": summarize([t / w for t, w in zip(tiles, walls)]),
                "seconds_per_tile": summarize([w / t for t, w in zip(tiles, walls) if t]),
                "device_seconds_per_cm2_gross": summarize([w / cm2(gross_px, cfg.pixel_um) for w in walls]),
                "device_seconds_per_cm2_net": summarize([w / cm2(net_px, cfg.pixel_um) for w in walls]) if net_px else summarize([]),
                "steady_state_tiles_per_second": summarize(steady) if steady else None,
                "stage_seconds": stage_seconds,
                "stage_coverage": coverage,
                "stage_breakdown_accepted": accepted,
                "area_pixels": {"gross": gross_px, "net": net_px},
            },
            "telemetry": {
                "per_repeat": per_rep_tel,
                "gpu_temperature_c_max": _agg("gpu_temperature_c_max", max),
                "gpu_sm_clock_mhz_min": _agg("gpu_sm_clock_mhz_min", min),
                "gpu_utilization_percent_avg": _agg("gpu_utilization_percent_avg", lambda v: sum(v) / len(v)),
                "peak_vram_bytes": _agg("peak_vram_bytes", max),
                "peak_host_rss_bytes": _agg("peak_host_rss_bytes", max),
            },
            "quality": {
                "output_sha256": hashes,
                "outputs_identical_across_repeats": len(set(hashes)) == 1 if hashes else None,
                "nonfinite_outputs": sum(nonfinite) if nonfinite else None,
                "note": "W0/W1 hashes are for determinism checks only; random weights say nothing about fp16 overflow",
            },
            "network": {k: v for k, v in no_upload_guard.network_report().items() if k != "attempts"},
        }
        errors = schema.validate(doc)
        if errors:
            raise RuntimeError("result does not match schema: " + "; ".join(errors))
        return doc
    finally:
        vars(inference.CFG).clear()
        vars(inference.CFG).update(saved_cfg)
        torch.use_deterministic_algorithms(saved_torch[0])
        torch.backends.cudnn.benchmark = saved_torch[1]
        shutil.rmtree(run_tmp, ignore_errors=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
