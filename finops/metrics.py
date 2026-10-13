"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """$/GB-VRAM-hour: what you pay per GB of model memory. 0 if hbm_gb<=0."""
    return on_demand_hr / hbm_gb if hbm_gb > 0 else 0.0


def recommend_rightsize(
    current_type: str,
    need_mem_gb: float,
    need_bw_tbs: float,
    need_tflops: float,
    catalog: dict,
    max_load: float = 0.70,
    mem_headroom: float = 1.10,
    max_mfu: float = 0.55,
    allowed_types=None,
) -> dict:
    """Cheapest catalog GPU that still fits a workload's measured demand.

    `need_*` are the workload's observed peaks (max memory, p95 bandwidth, p95 TFLOPs).
    A candidate must satisfy ALL of:
      - memory : hbm_gb >= need_mem_gb x mem_headroom      (weights + KV cache must fit)
      - bandwidth: need_bw / peak_bw <= max_load           (decode is bandwidth-bound)
      - compute: need_tflops / peak_tflops <= max_mfu     (MFU above ~0.55 is unreachable
        in practice, so a card that would need it is too small even if "on paper" it fits)
    `allowed_types` restricts candidates (e.g. same-vendor only: a cross-vendor move such as
    NVIDIA -> AMD MI300X also costs a software port, which $/GPU-hr does not show).
    Ranking is by $/GPU-hr among candidates that fit — the cheapest $/GPU-hr alone would
    pick a card that cannot hold the model or starves the bandwidth.
    `catalog` maps gpu_type -> dict with on_demand_hr, hbm_gb, peak_bw_tbs, peak_tflops_fp16.
    """
    def f(t, k):
        return float(catalog[t][k])

    fits = []
    for t in catalog:
        if allowed_types is not None and t not in allowed_types:
            continue
        if (f(t, "hbm_gb") >= need_mem_gb * mem_headroom
                and need_bw_tbs / f(t, "peak_bw_tbs") <= max_load
                and need_tflops / f(t, "peak_tflops_fp16") <= max_mfu):
            fits.append(t)
    # Current type may not even satisfy max_load (that's why we are looking) — it stays a fallback.
    best = min(fits, key=lambda t: f(t, "on_demand_hr")) if fits else current_type
    if f(best, "on_demand_hr") >= f(current_type, "on_demand_hr"):
        best = current_type
    cur_hr, new_hr = f(current_type, "on_demand_hr"), f(best, "on_demand_hr")
    return {
        "current": current_type, "recommended": best,
        "current_hr": cur_hr, "recommended_hr": new_hr,
        "savings_pct": round((1 - new_hr / cur_hr) * 100, 1) if cur_hr > 0 else 0.0,
        "new_mbu": round(need_bw_tbs / f(best, "peak_bw_tbs"), 3),
        "new_mfu": round(need_tflops / f(best, "peak_tflops_fp16"), 3),
    }


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)
