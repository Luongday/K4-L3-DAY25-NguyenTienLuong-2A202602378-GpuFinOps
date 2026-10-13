"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# --- Extension 1: interruption-aware, duration-aware tier policy -------------------------
# Illustrative per-hour spot reclaim probability by GPU type (assumption, not catalog data):
# scarce/premium parts are reclaimed less often than small inference parts.
INTERRUPTION_RATE_BY_GPU = {
    "H100": 0.03, "H200": 0.03, "A100": 0.04, "B200": 0.05,
    "MI300X": 0.06, "A10G": 0.08, "L4": 0.10,
}
# Commitment term in months; used to price stranded commitment when the service lives shorter.
RESERVED_TERM_MONTHS = {"reserved_1yr": 12, "reserved_3yr": 36}
# Lower number = lower lock-in. Used to break near-ties toward flexibility.
_COMMITMENT_RANK = {"on_demand": 0, "spot": 1, "reserved_1yr": 2, "reserved_3yr": 3}


def evaluate_tiers(
    hours_per_day: float,
    interruptible: bool,
    on_demand_hr: float,
    spot_hr: float,
    reserved_1yr_hr: float,
    reserved_3yr_hr: float,
    gpu_type: str | None = None,
    job_days: float = 30,
    lifetime_months: float = 24,
    days_per_month: int = 30,
) -> dict:
    """Monthly cost PER GPU of every feasible purchasing tier, with honest accounting.

    - on_demand: pay only for used hours.
    - spot: only if interruptible; includes checkpoint overhead + rework using the
      GPU-specific interruption rate (not a flat 5%).
    - reserved_*: a commitment bills every hour of the month (24h x days), used or not,
      so it is only eligible for steady services (job_days >= days_per_month). If the
      service is expected to live shorter than the term, the unused term is stranded:
      cost is scaled by term / lifetime.
    """
    used_hours = max(0.0, hours_per_day) * days_per_month
    options = {"on_demand": used_hours * on_demand_hr}
    if interruptible and hours_per_day < 24:
        rate = INTERRUPTION_RATE_BY_GPU.get(gpu_type or "", 0.05)
        sim = spot_checkpoint_cost(used_hours, spot_hr, on_demand_hr, interrupt_rate=rate)
        options["spot"] = sim["spot_cost"]
    if job_days >= days_per_month:  # steady-state service -> commitment is plausible
        full_month = 24.0 * days_per_month
        for tier, rate in (("reserved_1yr", reserved_1yr_hr), ("reserved_3yr", reserved_3yr_hr)):
            term = RESERVED_TERM_MONTHS[tier]
            stranded = max(1.0, term / max(lifetime_months, 1e-9))
            options[tier] = full_month * rate * stranded
    return options


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    on_demand_hr: float,
    spot_hr: float,
    reserved_1yr_hr: float,
    reserved_3yr_hr: float,
    gpu_type: str | None = None,
    job_days: float = 30,
    lifetime_months: float = 24,
    tie_tolerance: float = 0.01,
) -> dict:
    """Cost-minimizing tier. Among options within `tie_tolerance` of the cheapest, pick
    the least-committed one (flexibility is free option value)."""
    options = evaluate_tiers(hours_per_day, interruptible, on_demand_hr, spot_hr,
                             reserved_1yr_hr, reserved_3yr_hr, gpu_type, job_days, lifetime_months)
    cheapest = min(options.values())
    near = [t for t, c in options.items() if c <= cheapest * (1.0 + tie_tolerance)]
    tier = min(near, key=lambda t: _COMMITMENT_RANK[t])
    return {"tier": tier, "monthly_cost_per_gpu": options[tier], "options": options}


# --- Extension 3: is prompt caching actually worth it? ------------------------------------
def cache_breakeven_reads(write_cost_per_m: float = 0.25, read_discount: float = 0.10) -> float:
    """Reads of a cached prefix needed before caching beats not caching.

    Per cached token, relative to base input price p:
      without cache: p x (1 + n)                       (n re-reads, all full price)
      with cache   : p x (1 + w) + p x n x d           (one write premium w, reads at d)
    Saving = p x (n x (1 - d) - w)  ->  break-even n* = w / (1 - d).
    """
    saving_per_read = 1.0 - read_discount
    if saving_per_read <= 0:
        return float("inf")
    return max(0.0, write_cost_per_m) / saving_per_read


def cache_is_worth_it(avg_cache_reads: float, write_cost_per_m: float = 0.25,
                      read_discount: float = 0.10) -> bool:
    """True only if the average re-reads per cached prefix clear the break-even.

    `write_cost_per_m` is the EXTRA write cost as a multiple of the base input price
    (Anthropic 5-min write 1.25x -> 0.25; 1-hour TTL 2x -> 1.0; for storage-billed caches
    such as Gemini, fold the hourly storage fee into this number).
    """
    return avg_cache_reads > cache_breakeven_reads(write_cost_per_m, read_discount)


def cache_net_savings_per_m(avg_cache_reads: float, price_in_per_m: float,
                            write_cost_per_m: float = 0.25, read_discount: float = 0.10) -> float:
    """USD saved per 1M cached tokens over a prefix's life (negative = caching loses money)."""
    return price_in_per_m * (avg_cache_reads * (1.0 - read_discount) - write_cost_per_m)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
