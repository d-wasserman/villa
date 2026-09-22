"""
Command line interface: the contract between the three benchmark machines.

  python -m bench run --workload W1 --target-tag xps-1650 --mode throughput \\
      --batch-size 2 --workers 4 --compile reduce-overhead --inductor-cache cold \\
      --repeats 5 --seed 0 --out bench/results/xps-1650/
  python -m bench compare A1.json A2.json ... --b B1.json B2.json ...
  python -m bench validate FILE.json ...

Run from ink-detection/optimized_inference/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from bench import schema
from bench.harness import RunConfig, run
from bench.stats import compare_pairs

DEFAULT_METRIC = "timings.device_seconds_per_cm2_gross.mean"


def _add_run_args(p: argparse.ArgumentParser) -> None:
    d = RunConfig()
    p.add_argument("--workload", choices=("W0", "W1"), default=d.workload)
    p.add_argument("--target-tag", default=d.target_tag, help="machine label, e.g. claude-cpu, xps-1650, colab-t4")
    p.add_argument("--mode", choices=("throughput", "attribution"), default=d.mode)
    p.add_argument("--numerics", choices=("production", "reference"), default=d.numerics)
    p.add_argument("--autocast", choices=("auto", "on", "off"), default=d.autocast,
                   help="auto = on for CUDA (fp16), off for CPU (fp32). Ignored with --numerics reference")
    p.add_argument("--device", default=d.device, help="auto, cpu, cuda, cuda:1, ...")
    p.add_argument("--model", choices=("resnet3d-152-3d-decoder", "stub"), default=d.model)
    p.add_argument("--weights", default=d.weights, help="'random' or a local checkpoint path")
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--prefetch-factor", type=int, default=d.prefetch_factor)
    p.add_argument("--tile-size", type=int, default=d.tile_size)
    p.add_argument("--stride", type=int, default=d.stride)
    p.add_argument("--in-chans", type=int, default=d.in_chans)
    p.add_argument("--size", type=int, default=d.size, help="W1 synthetic volume side in pixels")
    p.add_argument("--w0-iters", type=int, default=d.w0_iters, help="W0 forward calls per repeat")
    p.add_argument("--compile", dest="compile_mode", choices=("off", "default", "reduce-overhead", "max-autotune"),
                   default=d.compile_mode)
    p.add_argument("--inductor-cache", choices=("cold", "warm"), default=d.inductor_cache)
    p.add_argument("--inductor-cache-dir", default=d.inductor_cache_dir,
                   help="warm cache location (default ~/.cache/villa-bench/inductor)")
    p.add_argument("--warmup", type=int, default=d.warmup, help="discarded repeats before timing")
    p.add_argument("--repeats", type=int, default=d.repeats)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--pixel-um", type=float, default=d.pixel_um, help="voxel size for the per-cm^2 metric")
    p.add_argument("--allow-host", dest="allow_hosts", action="append", default=[],
                   help="host the guard may contact (repeatable); W0/W1 need none")
    p.add_argument("--torch-trace", action="store_true", help="attribution mode: also run torch.profiler")
    p.add_argument("--out", default="", help="output directory (default bench/results/<target-tag>/)")
    p.add_argument("--tag", default="", help="suffix for the result file name")


def _get(doc, dotted: str):
    for part in dotted.split("."):
        doc = doc[part]
    return float(doc)


def cmd_run(args) -> int:
    fields = {k: v for k, v in vars(args).items() if k in RunConfig.__dataclass_fields__}
    cfg = RunConfig(**fields)
    doc = run(cfg)
    out_dir = Path(args.out or Path(__file__).resolve().parent / "results" / cfg.target_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = "-".join(p for p in (stamp, cfg.workload, cfg.mode, args.tag) if p) + ".json"
    path = out_dir / name
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    t = doc["timings"]
    print(json.dumps({
        "result": str(path),
        "seconds_per_tile_mean": t["seconds_per_tile"]["mean"],
        "cv": t["seconds_per_tile"]["cv"],
        "device_seconds_per_cm2_gross_mean": t["device_seconds_per_cm2_gross"]["mean"],
        "first_batch_seconds": t["first_batch_seconds"],
        "compile_seconds": t["compile_seconds"],
        "stage_coverage": t["stage_coverage"],
        "hosts_contacted": doc["network"]["hosts_contacted"],
    }, indent=2))
    return 0


def cmd_compare(args) -> int:
    a_docs = [json.loads(Path(p).read_text()) for p in args.a]
    b_docs = [json.loads(Path(p).read_text()) for p in args.b]
    for d in a_docs + b_docs:
        errs = schema.validate(d)
        if errs:
            raise SystemExit(f"invalid result file: {errs[:3]}")
    result = compare_pairs([_get(d, args.metric) for d in a_docs], [_get(d, args.metric) for d in b_docs],
                           confidence=args.confidence)
    result["metric"] = args.metric
    # Laptops throttle: flag runs whose minimum SM clock fell >10% below the best run's.
    clocks = {p: d["telemetry"]["gpu_sm_clock_mhz_min"] for p, d in zip(args.a + args.b, a_docs + b_docs)}
    known = [c for c in clocks.values() if c is not None]
    result["throttled_runs"] = [p for p, c in clocks.items() if c is not None and c < 0.9 * max(known)] if known else []
    if result["throttled_runs"]:
        result["warning"] = "rerun the pairs containing throttled runs before trusting the interval"
    print(json.dumps(result, indent=2))
    return 0


def cmd_flops(args) -> int:
    """Count FLOPs of one forward pass on the meta device (no weights, no compute)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    import model_resnet3d_3d_decoder as m3d

    with torch.device("meta"):
        net = m3d.RegressionModel(with_norm=False).eval()
        x = torch.empty((args.batch_size, 1, args.in_chans, args.tile_size, args.tile_size))
    counter = FlopCounterMode(display=False, depth=2)
    with torch.inference_mode(), counter:
        net(x)
    per_module = {name: sum(ops.values()) for name, ops in counter.get_flop_counts().items()}
    total = counter.get_total_flops()
    print(json.dumps({
        "input_shape": list(x.shape),
        "total_flops": total,
        "flops_per_tile": total / args.batch_size,
        "tflops_per_tile": total / args.batch_size / 1e12,
        "backbone_tflops_per_tile": per_module.get("RegressionModel.backbone", 0) / args.batch_size / 1e12,
        "decoder_tflops_per_tile": per_module.get("RegressionModel.decoder", 0) / args.batch_size / 1e12,
        "note": "FLOPs = 2 x multiply-adds, convolutions and matmuls only",
    }, indent=2))
    return 0


def cmd_validate(args) -> int:
    bad = 0
    for p in args.files:
        errs = schema.validate(json.loads(Path(p).read_text()))
        print(f"{p}: {'ok' if not errs else '; '.join(errs)}")
        bad += bool(errs)
    return 1 if bad else 0


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run one workload configuration")
    _add_run_args(p_run)
    p_run.set_defaults(func=cmd_run)
    p_cmp = sub.add_parser("compare", help="paired B/A ratio; give A runs, then --b B runs, in ABAB order")
    p_cmp.add_argument("a", nargs="+")
    p_cmp.add_argument("--b", nargs="+", required=True)
    p_cmp.add_argument("--metric", default=DEFAULT_METRIC, help=f"dotted path, default {DEFAULT_METRIC}")
    p_cmp.add_argument("--confidence", type=float, default=0.95)
    p_cmp.set_defaults(func=cmd_compare)
    p_fl = sub.add_parser("flops", help="FLOP count of one ResNet3D-152 decoder forward pass")
    p_fl.add_argument("--batch-size", type=int, default=1)
    p_fl.add_argument("--in-chans", type=int, default=62)
    p_fl.add_argument("--tile-size", type=int, default=256)
    p_fl.set_defaults(func=cmd_flops)
    p_val = sub.add_parser("validate", help="check result files against the schema")
    p_val.add_argument("files", nargs="+")
    p_val.set_defaults(func=cmd_validate)
    args = parser.parse_args(argv)
    sys.exit(args.func(args))
