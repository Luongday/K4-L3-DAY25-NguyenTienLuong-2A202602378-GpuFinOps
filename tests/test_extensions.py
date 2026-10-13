"""Tests for the "Your Turn" extensions (added by the student; the original tests are untouched)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, metrics, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, ext5_carbon_aware

# H100 row from the price catalog
H100 = dict(on_demand_hr=2.5, spot_hr=1.5, reserved_1yr_hr=2.0, reserved_3yr_hr=1.4)


# --- Extension 1 ---------------------------------------------------------------------
def test_tier_v2_prefers_1yr_when_service_lifetime_is_short():
    r = pricing.recommend_tier_v2(24, False, gpu_type="H100", lifetime_months=24, **H100)
    assert r["tier"] == "reserved_1yr"          # 3yr is stranded: 1.4 x 36/24 = 2.1 > 2.0
    r36 = pricing.recommend_tier_v2(24, False, gpu_type="H100", lifetime_months=36, **H100)
    assert r36["tier"] == "reserved_3yr"       # long-lived service amortizes the 3yr term


def test_tier_v2_never_commits_short_or_non_steady_jobs():
    assert pricing.recommend_tier_v2(24, False, job_days=5, gpu_type="H100", **H100)["tier"] == "on_demand"
    assert pricing.recommend_tier_v2(4, False, gpu_type="H100", **H100)["tier"] == "on_demand"
    assert pricing.recommend_tier_v2(4, True, gpu_type="H100", **H100)["tier"] == "spot"


def test_tier_v2_beats_or_matches_simple_policy_on_workloads():
    ext1 = m3_purchasing.run(verbose=False)["extension1"]
    assert ext1["v2_policy"] <= ext1["baseline_policy"]


# --- Extension 2 ---------------------------------------------------------------------
def test_rightsize_respects_memory_and_bandwidth():
    cat = m1_efficiency_audit.catalog_by_type()
    # 20 GB, tiny bandwidth, tiny compute: the cheapest card with >=22 GB wins
    rec = metrics.recommend_rightsize("A10G", 20.3, 0.18, 20, cat, allowed_types={"A10G", "L4", "H100"})
    assert rec["recommended"] == "L4"
    # 70 GB can never move to a 24 GB card even though it is cheaper per hour
    rec = metrics.recommend_rightsize("A100", 67.7, 0.69, 80, cat)
    assert rec["recommended"] == "A100" and rec["savings_pct"] == 0.0


# --- Extension 3 ---------------------------------------------------------------------
def test_cache_breakeven_and_gate():
    assert abs(pricing.cache_breakeven_reads(0.25, 0.10) - 0.25 / 0.9) < 1e-12
    assert pricing.cache_is_worth_it(1.0, 0.25) is True
    assert pricing.cache_is_worth_it(0.1, 0.25) is False
    assert pricing.cache_is_worth_it(1.0, 1.0) is False        # 1h TTL needs >1.11 reads
    assert pricing.cache_net_savings_per_m(0.0, 3.0, 0.25) < 0  # written, never read -> loss


def test_cache_extension_is_more_conservative_than_headline():
    r2 = m2_inference_levers.run(verbose=False)
    assert r2["cache_adjusted_pct"] <= r2["savings_pct"]


# --- Extension 4 ---------------------------------------------------------------------
def test_reasoning_budget_energy_dominates_cost():
    rx = m2_inference_levers.run(verbose=False)["reasoning_ext"]
    assert rx["wh_share"] > rx["usd_share"] > rx["traffic_share"]
    saved = [sc["usd_saved_daily"] for sc in rx["scenarios"]]
    assert saved == sorted(saved)                               # tighter cap never saves less
    assert rx["scenarios"][0]["usd_saved_daily"] == 0.0         # cap above current share is a no-op


# --- Extension 5 ---------------------------------------------------------------------
def test_carbon_aware_picks_expected_regions():
    r = ext5_carbon_aware.run(verbose=False)
    assert r["cleanest"] == "europe-north1"
    assert r["cheapest"] == "us-east-wa"
    assert r["saved_g_cleanest"] > 0 and r["saved_pct_cleanest"] > 80


def test_region_table_blends_cost_and_carbon():
    t = {x["region"]: x for x in sustainability.region_table(1000, 100.0)}
    assert abs(t["us-east-1"]["energy_usd"] - 120.0) < 1e-9
    assert abs(t["us-east-1"]["blended_usd"] - (120.0 + 380.0 * 1000 / 1e6 * 100.0)) < 1e-9
