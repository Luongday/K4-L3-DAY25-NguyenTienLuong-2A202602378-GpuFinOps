"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py

Extensions in this file: #3 cache_is_worth_it gate, #4 reasoning budget.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from datetime import datetime
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

CACHE_TTL_MIN = 5                                   # a prefix stays cached 5 min after last use
CACHE_WRITE_PREMIUM = {"5min": 0.25, "1h": 1.00}    # extra write cost x base input price
REASONING_CAPS = (0.10, 0.05, 0.02, 0.0)            # share of traffic allowed to use reasoning


def label_cache_events(rows):
    """Within each (route_tier, project) cache domain, in time order, label every request with
    a cached prefix as a WRITE (cold / TTL expired) or a READ (warm hit).

    Returns ({row_index: is_write}, {domain: {writes, reads, avg_reads}}).
    """
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["route_tier"], r["project"])].append((datetime.fromisoformat(r["ts"]), i))
    is_write, stats = {}, {}
    for key, items in groups.items():
        items.sort()
        last, writes, reads = None, 0, 0
        for t, i in items:
            if num(rows[i]["cached_input_tokens"]) <= 0:
                continue
            if last is None or (t - last).total_seconds() > CACHE_TTL_MIN * 60:
                is_write[i] = True
                writes += 1
            else:
                is_write[i] = False
                reads += 1
            last = t
        stats[key] = {"writes": writes, "reads": reads, "avg_reads": reads / writes if writes else 0.0}
    return is_write, stats


def cache_extension(rows) -> dict:
    """Extension 3: credit prompt caching only where re-reads clear the break-even, and bill the
    write premium that the headline number ignores."""
    w = CACHE_WRITE_PREMIUM["5min"]
    is_write, stats = label_cache_events(rows)
    worth = {k: pricing.cache_is_worth_it(v["avg_reads"], w) for k, v in stats.items()}
    adj = 0.0
    for i, r in enumerate(rows):
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        on = worth.get((r["route_tier"], r["project"]), False)
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached if on else 0)
        if on and is_write.get(i):  # the cold request pays the write premium on its prefix
            cost += cached / 1e6 * pin * (1.0 + w - 0.10)
        adj += cost * (0.5 if int(num(r["is_batch"])) else 1.0)
    by_tier = {}
    for tier, (pin, _) in MODEL_PRICES.items():
        vals = [v for (t, _p), v in stats.items() if t == tier]
        wr = sum(v["writes"] for v in vals)
        rd = sum(v["reads"] for v in vals)
        avg = rd / wr if wr else 0.0
        by_tier[tier] = {
            "observed_avg_reads": avg,
            "breakeven_5min": pricing.cache_breakeven_reads(CACHE_WRITE_PREMIUM["5min"]),
            "breakeven_1h": pricing.cache_breakeven_reads(CACHE_WRITE_PREMIUM["1h"]),
            "net_per_m_5min": pricing.cache_net_savings_per_m(avg, pin, CACHE_WRITE_PREMIUM["5min"]),
        }
    return {"groups": stats, "worth": worth, "adjusted_daily": adj, "by_tier": by_tier}


def reasoning_extension(rows) -> dict:
    """Extension 4: dollars and watt-hours of is_reasoning traffic, plus a routing-cap what-if.

    A capped request is re-served as a normal query: same input, the tier's average
    non-reasoning output length.
    """
    out_avg = {}
    for tier in MODEL_PRICES:
        outs = [num(r["output_tokens"]) for r in rows if r["route_tier"] == tier and r["is_reasoning"] == "0"]
        out_avg[tier] = sum(outs) / len(outs)
    tot = {"n": 0, "usd": 0.0, "wh": 0.0}
    rs = {"n": 0, "usd": 0.0, "wh": 0.0, "usd_alt": 0.0, "wh_alt": 0.0}
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        batch = bool(int(num(r["is_batch"])))
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=batch)
        reasoning = r["is_reasoning"] == "1"
        wh = sustainability.wh_per_query(inp + out, is_reasoning=reasoning)
        tot["n"] += 1
        tot["usd"] += cost
        tot["wh"] += wh
        if reasoning:
            alt_out = int(out_avg[r["route_tier"]])
            rs["n"] += 1
            rs["usd"] += cost
            rs["wh"] += wh
            rs["usd_alt"] += pricing.request_cost(inp, alt_out, pin, pout, cached_in=cached, batch=batch)
            rs["wh_alt"] += sustainability.wh_per_query(inp + alt_out)
    share = rs["n"] / tot["n"]
    scenarios = []
    for cap in REASONING_CAPS:
        f = max(0.0, 1.0 - cap / share) if share else 0.0  # fraction of reasoning traffic re-routed
        scenarios.append({"cap": cap, "rerouted_frac": f,
                          "usd_saved_daily": f * (rs["usd"] - rs["usd_alt"]),
                          "wh_saved_daily": f * (rs["wh"] - rs["wh_alt"])})
    return {"traffic_share": share, "usd_share": rs["usd"] / tot["usd"], "wh_share": rs["wh"] / tot["wh"],
            "usd": rs["usd"], "wh": rs["wh"], "total_usd": tot["usd"], "total_wh": tot["wh"],
            "scenarios": scenarios}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    cache_x = cache_extension(rows)
    reason_x = reasoning_extension(rows)
    adj_pct = (1 - cache_x["adjusted_daily"] / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Extension 3: is caching worth it? (break-even reads = write premium / (1 - read price)) --")
        for tier, v in cache_x["by_tier"].items():
            print(f"{tier:6} observed re-reads/prefix={v['observed_avg_reads']:.1f}  "
                  f"break-even: 5min-TTL={v['breakeven_5min']:.2f}  1h-TTL={v['breakeven_1h']:.2f}  "
                  f"net=${v['net_per_m_5min']:.2f} per 1M cached tokens")
        bad = [k for k, ok in cache_x["worth"].items() if not ok]
        print(f"cache domains below break-even: {bad or 'none'}")
        print(f"optimized incl. write premium & gate: ${cache_x['adjusted_daily']:.2f}/day  "
              f"({adj_pct:.1f}% vs baseline; headline {savings_pct:.1f}%)")

        r = reason_x
        print("\n-- Extension 4: reasoning budget --")
        print(f"reasoning = {r['traffic_share']:.1%} of requests, {r['usd_share']:.1%} of $ "
              f"(${r['usd']:.2f}/day), {r['wh_share']:.1%} of energy ({r['wh']:,.0f} of {r['total_wh']:,.0f} Wh/day)")
        for sc in r["scenarios"]:
            print(f"  cap {sc['cap']:>4.0%} of traffic: re-route {sc['rerouted_frac']:.0%} of reasoning -> "
                  f"save ${sc['usd_saved_daily']:.2f}/day (${sc['usd_saved_daily'] * 30:,.0f}/mo), "
                  f"{sc['wh_saved_daily']:,.0f} Wh/day")
        print("  rule: reasoning only for offline/batch jobs AND (task complexity high OR small-model "
              "confidence < threshold), under a per-project reasoning budget")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_ext": cache_x, "reasoning_ext": reason_x, "cache_adjusted_pct": round(adj_pct, 1),
    }


if __name__ == "__main__":
    run()
