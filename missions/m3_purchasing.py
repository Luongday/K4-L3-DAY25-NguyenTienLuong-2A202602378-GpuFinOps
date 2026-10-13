"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30
LIFETIME_MONTHS = 24   # assumed remaining life of a steady service (drives 1yr vs 3yr)
SHORT = {"on_demand": "OD", "spot": "SPOT", "reserved_1yr": "R1", "reserved_3yr": "R3"}


def _rates(c):
    return (num(c["on_demand_hr"]), num(c["spot_hr"]),
            num(c["reserved_1yr_hr"]), num(c["reserved_3yr_hr"]))


def _honest_cost(tier, hpd, interruptible, c, gpu_type, job_days, ngpu):
    """Monthly cost of a tier under the SAME honest accounting for any policy
    (reserved bills 24h x 30d whether used or not; spot uses the GPU's own reclaim rate)."""
    od, spot, r1, r3 = _rates(c)
    opts = pricing.evaluate_tiers(hpd, interruptible, od, spot, r1, r3, gpu_type, job_days, LIFETIME_MONTHS)
    return opts[tier] * ngpu


def extension1(jobs, cat) -> dict:
    """Interruption- and duration-aware policy vs the simple baseline policy."""
    rows, od_total, base_total, v2_total = [], 0.0, 0.0, 0.0
    for j in jobs:
        gtype, ngpu = j["gpu_type"], int(num(j["num_gpus"]))
        hpd, jdays = num(j["hours_per_day"]), num(j["days"])
        intr = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        od, spot, r1, r3 = _rates(c)
        base_tier = pricing.recommend_tier(hpd, intr)
        base_key = "reserved_3yr" if base_tier == "reserved" else base_tier  # baseline implies 3yr rates
        v2 = pricing.recommend_tier_v2(hpd, intr, od, spot, r1, r3, gtype, jdays, LIFETIME_MONTHS)
        od_cost = hpd * DAYS * ngpu * od
        base_cost = od_cost if (base_key == "reserved_3yr" and jdays < DAYS) else             _honest_cost(base_key, hpd, intr, c, gtype, jdays, ngpu)
        v2_cost = v2["monthly_cost_per_gpu"] * ngpu
        od_total += od_cost; base_total += base_cost; v2_total += v2_cost
        rows.append({"job_id": j["job_id"], "gpu_type": gtype, "base_tier": base_key, "v2_tier": v2["tier"],
                     "base_cost": round(base_cost), "v2_cost": round(v2_cost),
                     "reclaim_rate": pricing.INTERRUPTION_RATE_BY_GPU.get(gtype)})
    return {"rows": rows, "on_demand": round(od_total), "baseline_policy": round(base_total),
            "v2_policy": round(v2_total),
            "baseline_pct": round((1 - base_total / od_total) * 100, 1),
            "v2_pct": round((1 - v2_total / od_total) * 100, 1)}


def tier_matrix(cat) -> list:
    """GPU type x duty-hours x interruptible -> chosen tier (job_days>=30, steady)."""
    out = []
    for gtype, c in cat.items():
        od, spot, r1, r3 = _rates(c)
        cells = []
        for hpd in (4, 12, 18, 24):
            for intr in (True, False):
                t = pricing.recommend_tier_v2(hpd, intr, od, spot, r1, r3, gtype, 30, LIFETIME_MONTHS)["tier"]
                cells.append(SHORT[t])
        out.append((gtype, cells))
    return out


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

    ext1 = extension1(jobs, cat)
    if verbose:
        print("\n-- Extension 1: interruption- & duration-aware tier policy --")
        print("(both policies re-costed with honest accounting: reserved bills every hour of the month)")
        print(f"{'job':18}{'gpu':7}{'reclaim':>8}  {'baseline':14}{'$':>8}  {'v2':14}{'$':>8}")
        for r in ext1["rows"]:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['reclaim_rate']:>8.0%}  {r['base_tier']:14}{r['base_cost']:>8,}  "
                  f"{r['v2_tier']:14}{r['v2_cost']:>8,}")
        print(f"on-demand ${ext1['on_demand']:,} | simple policy ${ext1['baseline_policy']:,} ({ext1['baseline_pct']}%) "
              f"| v2 policy ${ext1['v2_policy']:,} ({ext1['v2_pct']}%)")
        print("\ntier matrix (steady service, lifetime 24 mo)  cols = 4h 12h 18h 24h x (interruptible Y/N)")
        print(f"{'gpu':8}" + "".join(f"{h + ('Y' if y else 'N'):>7}" for h in ('4h', '12h', '18h', '24h') for y in (True, False)))
        for g, cells in tier_matrix(cat):
            print(f"{g:8}" + "".join(f"{c:>7}" for c in cells))

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "extension1": ext1}


if __name__ == "__main__":
    run()
