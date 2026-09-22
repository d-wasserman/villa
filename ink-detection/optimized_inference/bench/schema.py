"""
results.json schema (version 1) and a plain-Python validator.

Every machine (Claude's CPU container, the GTX 1650, Colab T4) writes this
same shape, so runs can be compared side by side. ``validate`` returns a
list of error strings; an empty list means the document is valid.
"""
from __future__ import annotations

from typing import Any, Dict, List

SCHEMA_VERSION = 1

NoneType = type(None)
NUM = (int, float)
OPT_NUM = (int, float, NoneType)
OPT_STR = (str, NoneType)
OPT_INT = (int, NoneType)
OPT_BOOL = (bool, NoneType)


class ListOf:
    def __init__(self, item_types, optional: bool = False):
        self.item_types = item_types
        self.optional = optional

    def check(self, value: Any) -> bool:
        if value is None:
            return self.optional
        return isinstance(value, list) and all(_is(v, self.item_types) for v in value)

    def __repr__(self) -> str:
        return f"list[{self.item_types}]"


class DictType:
    def __init__(self, optional: bool = False):
        self.optional = optional

    def check(self, value: Any) -> bool:
        return (value is None and self.optional) or isinstance(value, dict)

    def __repr__(self) -> str:
        return "dict"


def _is(value: Any, types) -> bool:
    if isinstance(types, (ListOf, DictType)):
        return types.check(value)
    # bool is a subclass of int; only accept it where bool is listed
    if isinstance(value, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
        return False
    return isinstance(value, types)


SUMMARY_FIELDS = {"n": int, "mean": OPT_NUM, "p50": OPT_NUM, "p95": OPT_NUM, "min": OPT_NUM, "max": OPT_NUM, "cv": OPT_NUM}

SPEC: Dict[str, Dict[str, Any]] = {
    "identity": {
        "schema_version": int,
        "run_id": str,
        "created_utc": str,
        "git_commit": OPT_STR,
        "git_dirty": OPT_BOOL,
        "target_tag": str,
    },
    "environment": {
        "device": str,
        "gpu_name": OPT_STR,
        "gpu_compute_capability": OPT_STR,
        "gpu_driver": OPT_STR,
        "torch_version": str,
        "cuda_version": OPT_STR,
        "cudnn_version": OPT_INT,
        "os": str,
        "machine": str,
        "python_version": str,
        "cpu_model": OPT_STR,
        "ram_bytes": OPT_INT,
        "cpu_count_visible": int,
    },
    "config": {
        "workload": str,
        "mode": str,
        "numerics": str,
        "autocast": bool,
        "deterministic": bool,
        "compile_mode": str,
        "inductor_cache": str,
        "inductor_cache_state": str,
        "model": str,
        "weights": str,
        "weights_sha256": OPT_STR,
        "seed": int,
        "batch_size": int,
        "workers": int,
        "prefetch_factor": int,
        "tile_size": int,
        "stride": int,
        "in_chans": int,
        "input_shape": ListOf(int),
        "pixel_um": NUM,
        "warmup": int,
        "repeats": int,
        "crop_ids": ListOf(str),
        "cfg": DictType(),
    },
    "timings": {
        "model_build_seconds": NUM,
        "compile_seconds": OPT_NUM,
        "first_batch_seconds": OPT_NUM,
        "repeat_wall_seconds": ListOf(NUM),
        "tiles_total": int,
        "tiles_forwarded": ListOf(int),
        "tiles_per_second": DictType(),
        "seconds_per_tile": DictType(),
        "device_seconds_per_cm2_gross": DictType(),
        "device_seconds_per_cm2_net": DictType(),
        "steady_state_tiles_per_second": DictType(optional=True),
        "stage_seconds": DictType(optional=True),
        "stage_coverage": OPT_NUM,
        "stage_breakdown_accepted": OPT_BOOL,
    },
    "telemetry": {
        "per_repeat": ListOf(dict),
        "gpu_temperature_c_max": OPT_NUM,
        "gpu_sm_clock_mhz_min": OPT_NUM,
        "gpu_utilization_percent_avg": OPT_NUM,
        "peak_vram_bytes": OPT_INT,
        "peak_host_rss_bytes": OPT_INT,
    },
    "quality": {
        "output_sha256": ListOf(str),
        "outputs_identical_across_repeats": OPT_BOOL,
        "nonfinite_outputs": OPT_INT,
    },
    "network": {
        "guard_active": bool,
        "allow_hosts": ListOf(str),
        "hosts_contacted": ListOf(str),
        "hosts_blocked": ListOf(str),
    },
}

SUMMARY_KEYS = ("tiles_per_second", "seconds_per_tile", "device_seconds_per_cm2_gross", "device_seconds_per_cm2_net", "steady_state_tiles_per_second")
ALLOWED_VALUES = {
    ("config", "mode"): ("throughput", "attribution"),
    ("config", "numerics"): ("production", "reference"),
    ("config", "workload"): ("W0", "W1"),
    ("config", "inductor_cache"): ("cold", "warm", "n/a"),
}


def validate(doc: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(doc, dict):
        return ["document is not an object"]
    for group, fields in SPEC.items():
        section = doc.get(group)
        if not isinstance(section, dict):
            errors.append(f"missing group '{group}'")
            continue
        for name, types in fields.items():
            if name not in section:
                errors.append(f"missing field '{group}.{name}'")
            elif not _is(section[name], types):
                errors.append(f"field '{group}.{name}' has wrong type: {type(section[name]).__name__}, expected {types}")
    if not errors:
        if doc["identity"]["schema_version"] != SCHEMA_VERSION:
            errors.append(f"schema_version {doc['identity']['schema_version']} != {SCHEMA_VERSION}")
        for (group, name), allowed in ALLOWED_VALUES.items():
            if doc[group][name] not in allowed:
                errors.append(f"field '{group}.{name}' = {doc[group][name]!r} not in {allowed}")
        for key in SUMMARY_KEYS:
            summary = doc["timings"][key]
            if summary is None:
                continue
            for field, types in SUMMARY_FIELDS.items():
                if field not in summary or not _is(summary[field], types):
                    errors.append(f"summary 'timings.{key}.{field}' missing or wrong type")
        n = len(doc["timings"]["repeat_wall_seconds"])
        if n != doc["config"]["repeats"]:
            errors.append(f"{n} repeat timings for repeats={doc['config']['repeats']}")
        if len(doc["timings"]["tiles_forwarded"]) != n or len(doc["telemetry"]["per_repeat"]) != n:
            errors.append("per-repeat lists have inconsistent lengths")
    return errors
