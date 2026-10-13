"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


UNDERUSED_MFU = 0.35   # below the "good" MFU band
UNDERUSED_MBU = 0.50   # and not saturating HBM either -> over-provisioned card
NON_NVIDIA = {"MI300X"}  # cross-vendor moves need a ROCm port; reported separately


def _p95(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0,
                               "tflops": [], "bw": [], "mem": [], "workload": None})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        a["tflops"].append(num(r["achieved_tflops"]))
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["mem"].append(num(r["mem_used_gb"]))
        a["workload"] = r["workload"]
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # --- Extension 2: right-size under-used GPUs by measured memory / bandwidth / compute ---
    rightsizing = []
    for gid, a in agg.items():
        s = next(x for x in summary if x["gpu_id"] == gid)
        if s["mfu"] >= UNDERUSED_MFU or s["mbu"] >= UNDERUSED_MBU:
            continue  # healthy GPU: leave alone
        need = (max(a["mem"]), _p95(a["bw"]), _p95(a["tflops"]))
        rec = metrics.recommend_rightsize(a["type"], *need, cat,
                                          allowed_types=set(cat) - NON_NVIDIA)
        anyv = metrics.recommend_rightsize(a["type"], *need, cat)
        rec.update({"gpu_id": gid, "workload": a["workload"], "mbu": s["mbu"], "mfu": s["mfu"],
                    "need_mem_gb": round(need[0], 1), "need_bw_tbs": round(need[1], 3),
                    "monthly_savings": round((rec["current_hr"] - rec["recommended_hr"]) * 24 * 30),
                    "any_vendor": anyv["recommended"],
                    "any_vendor_savings": round((anyv["current_hr"] - anyv["recommended_hr"]) * 24 * 30)})
        rightsizing.append(rec)
    rightsize_monthly = sum(r["monthly_savings"] for r in rightsizing)
    rightsize_any_vendor = sum(r["any_vendor_savings"] for r in rightsizing)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- Extension 2: right-sizing by measured demand (MFU<0.35 & MBU<0.50) --")
        print("$/GB-VRAM-hr:  " + "  ".join(
            f"{t}=${metrics.dollars_per_gb_vram(num(c['on_demand_hr']), num(c['hbm_gb'])):.3f}"
            for t, c in cat.items()))
        print(f"{'GPU':13}{'kind':7}{'MBU':>6}{'mem_GB':>8}{'bw_p95':>8}  {'now':7}{'->':3}{'NVIDIA rec':11}{'new MBU':>8}{'save%':>7}{'$/mo':>7}  any-vendor")
        for r in rightsizing:
            print(f"{r['gpu_id']:13}{r['workload']:7}{r['mbu']:>6}{r['need_mem_gb']:>8}{r['need_bw_tbs']:>8}  "
                  f"{r['current']:7}{'->':3}{r['recommended']:11}{r['new_mbu']:>8}{r['savings_pct']:>7}{r['monthly_savings']:>7,}  "
                  f"{r['any_vendor']} (${r['any_vendor_savings']:,}/mo)")
        print(f"right-sizing total: ${rightsize_monthly:,}/month NVIDIA-only; "
              f"${rightsize_any_vendor:,}/month if a ROCm port to MI300X is acceptable")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsizing": rightsizing, "rightsize_monthly": rightsize_monthly,
            "rightsize_any_vendor": rightsize_any_vendor}


if __name__ == "__main__":
    run()
