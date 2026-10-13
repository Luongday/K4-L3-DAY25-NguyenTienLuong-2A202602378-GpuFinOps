"""Extension 5 — Carbon-aware scheduling of interruptible jobs.

Run: python missions/ext5_carbon_aware.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

HOME_REGION = "us-east-1"
PUE = 1.2                      # datacenter overhead on top of GPU board power
CARBON_PRICE_USD_PER_T = 100.0  # internal carbon price used to blend $ and CO2


def run(verbose: bool = True) -> dict:
    cat = catalog_by_type()
    jobs = [j for j in load_csv("workloads.csv") if int(num(j["interruptible"])) == 1]

    per_job, total_kwh, gpu_rent = [], 0.0, 0.0
    for j in jobs:
        c = cat[j["gpu_type"]]
        gpu_hours = num(j["hours_per_day"]) * num(j["days"]) * num(j["num_gpus"])
        kwh = gpu_hours * num(c["watts"]) / 1000.0 * PUE
        rent = gpu_hours * num(c["spot_hr"])
        total_kwh += kwh
        gpu_rent += rent
        home_g = sustainability.carbon_g(kwh * 1000, HOME_REGION)
        best_g = sustainability.carbon_g(kwh * 1000, min(sustainability.REGION_CARBON,
                                                         key=sustainability.REGION_CARBON.get))
        per_job.append({"job_id": j["job_id"], "gpu_hours": gpu_hours, "kwh": kwh,
                        "home_g": home_g, "clean_g": best_g})

    table = sustainability.region_table(total_kwh, CARBON_PRICE_USD_PER_T)
    cleanest = min(table, key=lambda r: r["g_per_kwh"])
    cheapest = min(table, key=lambda r: r["energy_usd"])
    balanced = min(table, key=lambda r: r["blended_usd"])
    home = next(r for r in table if r["region"] == HOME_REGION)

    saved_g = home["carbon_g"] - cleanest["carbon_g"]
    result = {
        "jobs": per_job, "total_kwh": total_kwh, "table": table, "home": home,
        "cleanest": cleanest["region"], "cheapest": cheapest["region"], "balanced": balanced["region"],
        "saved_g_cleanest": saved_g, "saved_pct_cleanest": saved_g / home["carbon_g"] * 100,
        "saved_g_balanced": home["carbon_g"] - balanced["carbon_g"],
        "saved_usd_balanced": home["energy_usd"] - balanced["energy_usd"],
        "gpu_spot_rent": gpu_rent,
    }

    if verbose:
        print("== Extension 5: carbon-aware scheduling (interruptible jobs only) ==")
        print(f"{'job':18}{'GPU-h':>9}{'kWh':>9}{HOME_REGION + ' gCO2':>16}{'cleanest gCO2':>15}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['gpu_hours']:>9,.0f}{p['kwh']:>9,.0f}{p['home_g']:>16,.0f}{p['clean_g']:>15,.0f}")
        print(f"total energy: {total_kwh:,.0f} kWh  (PUE {PUE})\n")
        print(f"{'region':17}{'$/kWh':>7}{'gCO2/kWh':>10}{'elec $':>9}{'CO2 kg':>9}{'blended $':>11}")
        for r in table:
            print(f"{r['region']:17}{r['usd_per_kwh']:>7.3f}{r['g_per_kwh']:>10}{r['energy_usd']:>9.2f}"
                  f"{r['carbon_g'] / 1000:>9.1f}{r['blended_usd']:>11.2f}")
        print(f"\ncheapest ($): {result['cheapest']} | cleanest (CO2): {result['cleanest']} | "
              f"balanced (@ ${CARBON_PRICE_USD_PER_T:.0f}/tCO2): {result['balanced']}")
        print(f"moving all interruptible jobs {HOME_REGION} -> {result['cleanest']}: "
              f"-{saved_g / 1000:,.1f} kgCO2e ({result['saved_pct_cleanest']:.0f}%), "
              f"electricity ${home['energy_usd']:.2f} -> ${cleanest['energy_usd']:.2f}")
        print(f"electricity is only {home['energy_usd'] / gpu_rent:.1%} of the GPU spot rent "
              f"(${gpu_rent:,.0f}) -> carbon can drive the choice almost for free; "
              "the real constraints are GPU availability/spot price per region, data egress and latency "
              "(fine for training, bad for user-facing inference).")
    return result


if __name__ == "__main__":
    run()
