# Benchmark harness (2 µm ink inference)

A profiling harness for `STEP=inference` with `MODEL_TYPE=resnet3d-152-3d-decoder`.
It calls the production code directly (`inference.configure_runtime`,
`inference.run_inference`). It never calls `entrypoint.py` step functions and
cannot upload anything. Every machine runs the same CLI and writes the same
`results.json` schema, so numbers can be compared side by side.

Run everything from `ink-detection/optimized_inference/`.

## Install

```bash
pip install torch==2.10.*            # CUDA build on GPU machines; CPU wheel elsewhere
pip install -r requirements-cpu-only.txt setuptools   # setuptools: Inductor on CPU
```

## Commands

```bash
# W0: forward pass only, random tensor (per-tile cost, VRAM)
python -m bench run --workload W0 --target-tag xps-1650 --batch-size 2 --w0-iters 5

# W1: full inference loop on a synthetic 1024x1024x62 volume
python -m bench run --workload W1 --target-tag xps-1650 --mode throughput \
    --batch-size 2 --workers 4 --compile reduce-overhead --inductor-cache cold \
    --repeats 5 --seed 0

# Where does time go? (CUDA syncs per stage; slower)
python -m bench run --workload W1 --target-tag xps-1650 --mode attribution

# A/B: run A and B alternately (ABAB, >= 5 pairs), then
python -m bench compare A1.json A2.json A3.json A4.json A5.json --b B1.json B2.json B3.json B4.json B5.json

python -m bench flops                  # FLOPs per tile (meta device, instant)
python -m bench validate FILE.json     # schema check
```

Results go to `bench/results/<target-tag>/<timestamp>-<workload>-<mode>[-<tag>].json`
unless `--out` is given. They are small JSON summaries, never prediction arrays.

## Run matrix per machine

**GTX 1650 laptop (Ubuntu 24.04).** Plug in and let the GPU cool between
runs. Batch 4 may not fit in 4 GB with CUDA graphs.

```bash
for bs in 1 2; do
  python -m bench run --workload W0 --target-tag xps-1650 --batch-size $bs --w0-iters 5 --tag bs$bs
done
for c in off default reduce-overhead; do
  for w in 0 2 4; do
    python -m bench run --workload W1 --target-tag xps-1650 --batch-size 2 --workers $w \
        --compile $c --inductor-cache cold --tag c-$c-w$w
  done
  python -m bench run --workload W1 --target-tag xps-1650 --mode attribution --batch-size 2 \
      --workers 4 --compile $c --tag c-$c
done
# warm-cache compile time: run the same config twice with --inductor-cache warm
```

**Colab T4.** Open `bench/colab.ipynb`, select a T4 runtime and run all
cells. It sweeps W0 over batch 1–16 until out of memory, runs W1 with
compile off and reduce-overhead, and zips the result JSONs for download.

**Claude CPU container.** Use `--size 512` for W1; a tile takes several
seconds on CPU.

## Options that matter

| Option | Values | Notes |
|---|---|---|
| `--model` | `resnet3d-152-3d-decoder`, `stub` | `stub` is a tiny conv net with the same input and output shapes, for plumbing tests |
| `--weights` | `random`, a checkpoint path | Random weights run at the same GPU speed as real ones, so W0 and W1 need no download. With a path, any missing or unexpected key fails the run. Production `load_model` only logs these |
| `--numerics` | `production`, `reference` | Reference: fp32, `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark` off, no compile |
| `--autocast` | `auto`, `on`, `off` | `auto` is fp16 on CUDA and fp32 on CPU. CPU autocast would be bf16, which production never runs |
| `--compile` | `off`, `default`, `reduce-overhead`, `max-autotune` | Same `configure_runtime` as production, including its batch-1 warmup |
| `--inductor-cache` | `cold`, `warm` | Cold uses a fresh temporary cache. Warm uses `~/.cache/villa-bench/inductor`, and the result records whether it was already populated |
| `--size` | pixels | W1 volume side. 1024 gives 49 tiles; use 512 (9 tiles) on CPU |

## Timing modes

Timing modes are never mixed in one result file.

- **throughput**: no extra CUDA syncs, as in production. Wall time per repeat
  covers the whole `run_inference` call: loader and worker start, the loop,
  the zarr write and teardown. This feeds the headline metric.
- **attribution**: the profiler's `detailed` level, which syncs CUDA per stage.
  `torch.profiler` stays off unless `--torch-trace` is passed, because it would
  distort the stage times. The measured stages are setup, loader start,
  loader wait, host-to-device, forward, device-to-host, postprocess, zarr write
  and teardown. They must add up to within 10% of wall time
  (`stage_breakdown_accepted`).

Every run discards `--warmup` repeats, then reports mean, p50, p95, min, max
and CV over `--repeats`. The first batch of the first repeat is reported
separately (`first_batch_seconds`): it includes worker spawn and the
recompile at the real batch size. Compile time is also reported separately
(`compile_seconds`).

## Headline metric: device-seconds per cm²

`device_seconds_per_cm2_gross` divides by the whole grid footprint.
`device_seconds_per_cm2_net` divides by the pixels that hold data (any nonzero
layer). They differ when a segment has empty areas, as W1 does by design,
and an empty-tile optimization moves the net figure more than the gross one.
`--pixel-um` sets the voxel size (default 2.4). For W0, each tile counts as
stride² pixels, the new area one tile adds to a fully covered segment
(about 1,060 tiles per cm² at stride 128).

`compare` uses the t-interval on log(B/A) as its decision rule. It also
prints a bootstrap interval, which is too narrow with few pairs. Runs whose
minimum SM clock dropped more than 10% below the best run's are listed under
`throttled_runs`; rerun those pairs.

## Safety guard

Importing `bench` installs `bench/no_upload_guard.py`:

- `boto3.client` and `boto3.resource` raise.
- s3fs write modes and mutating methods raise.
- Every outbound socket connection must be loopback or on the `--allow-host`
  list. A configured HTTP(S) proxy counts as external. W0 and W1 allow nothing.
- Spawned DataLoader workers get the guard through `worker_init_fn`.

Every result file records the hosts contacted and blocked under `network`.
The harness also sets `NO_ALBUMENTATIONS_UPDATE=1`. Without it, albumentations
contacts PyPI on every import, including in each spawned worker.

## Tests

```bash
python -m unittest discover -s tests -t .
```

`tests/test_bench_guard.py` covers the guard, including inside a spawned
worker. `tests/test_bench.py` runs W0 and W1 end to end with the stub model
and checks the schema, determinism, and that global state is restored
(`inference.CFG`, the loader factory, torch determinism flags).
