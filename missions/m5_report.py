"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, ext5_carbon_aware

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def _analysis_sections(r1, r2, r3, r5x, levers, baseline) -> list:
    """Vietnamese analysis blocks: util-lie mechanism, prioritized actions, extension results."""
    lies = ", ".join(f"`{l['gpu_id']}` ({l['gpu_type']}, util {l['gpu_util_pct']}%, MFU {l['mfu']})"
                     for l in r1["lies"])
    ranked = sorted(levers.items(), key=lambda kv: -kv[1])
    prio = "\n".join(f"{i}. **{n}** — ${v:,.0f}/tháng ({v / baseline * 100:.1f}% baseline)"
                      for i, (n, v) in enumerate(ranked, 1))
    e1, c, rx = r3["extension1"], r2["cache_ext"], r2["reasoning_ext"]
    rs_total = sum(x["monthly_savings"] for x in r1["rightsizing"])
    cap5 = next(sc for sc in rx["scenarios"] if sc["cap"] == 0.05)
    bal = next(t for t in r5x["table"] if t["region"] == r5x["balanced"])

    lie_block = f"""## GPU-Util "nói dối" — cơ chế và tác động tài chính

GPU phát hiện: {lies}.

`nvidia-smi` GPU-Util chỉ là tỷ lệ thời gian *có ít nhất một kernel đang chạy*, không đo số FLOPs sinh ra. Một kernel nhỏ, bị chờ HBM (memory stall), chờ dữ liệu từ dataloader/I/O, hoặc launch overhead vẫn giữ util ở ~98% trong khi tensor core gần như rảnh. Vì vậy MFU chỉ ~0.2 (so với 0.40–0.43 ở các H100 khỏe cùng cụm): ta trả đủ giờ GPU nhưng nhận ~1/2 sản lượng của một H100 bình thường. Tác động: mỗi H100 bị lie đốt ~${(2.5 * 24 * 30):,.0f}/tháng on-demand cho ~50% công việc hữu ích, tương đương ~${(2.5 * 24 * 30) / 2:,.0f}/tháng lãng phí/GPU. Cách xử lý đúng là sửa nút thắt (dataloader, batch size, kernel fusion), không phải mua thêm GPU — right-sizing theo số đo (Extension 2) cho thấy đổi phần cứng chỉ tiết kiệm được ${rs_total:,}/tháng khi giữ NVIDIA."""

    prio_block = f"""## Hành động ưu tiên (theo ROI)

{prio}

Bảng trên xếp theo giá trị tiết kiệm tuyệt đối. Thứ tự thực thi đề xuất (theo rủi ro và thời gian hoàn vốn): (1) chuyển job gián đoạn được sang spot có checkpoint — lever lớn nhất, không phải cam kết; (2) cascade + batch + caching cho inference — giảm 83% $/1M-token, không rủi ro và tăng giá trị theo lưu lượng (hiện chỉ 2,400 request/ngày nên số tiền tuyệt đối còn nhỏ); (3) reserved 1 năm cho dịch vụ chạy 24/7 sau khi đo duty cycle thật; (4) điều tra dataloader/pipeline của các GPU "lie" trước khi right-size; (5) tắt GPU idle."""

    ext_block = f"""## Kết quả các phần mở rộng "Your Turn"

**Ext 1 — Chính sách tier theo tỷ lệ gián đoạn & thời hạn cam kết.** Tính lại cả hai chính sách với cùng cách tính trung thực (reserved bị tính đủ 24h×30 ngày dù dùng hay không; spot dùng tỷ lệ reclaim riêng từng GPU): chính sách đơn giản tiết kiệm {e1['baseline_pct']}% so với on-demand (${e1['baseline_policy']:,}), chính sách mới {e1['v2_pct']}% (${e1['v2_policy']:,}). (Lưu ý: con số 26.4% thấp hơn headline 39.1% của M3 vì M3 chỉ tính reserved cho giờ sử dụng, còn thực tế reserved bị tính đủ 24h — headline M3 đang lạc quan với dịch vụ chạy <24h/ngày.) Khác biệt giữa hai chính sách đến từ: reserved 1 năm thay vì 3 năm khi vòng đời dịch vụ ước tính 24 tháng (3 năm bị "stranded"), và `job-infer-search` (L4, 18h/ngày) nên giữ on-demand vì reserved không rẻ hơn khi tính đủ giờ.

**Ext 2 — Right-sizing theo MBU/bộ nhớ/compute.** Chọn GPU rẻ nhất vẫn đủ VRAM (×1.1), băng thông (≤70% peak) và compute (≤0.55 MFU). Chỉ `gpu-a10g-0` đổi A10G→L4 (-20%, $144/tháng); tổng ${rs_total:,}/tháng với NVIDIA, ${r1['rightsize_any_vendor']:,}/tháng nếu chấp nhận port sang MI300X. Không chọn theo `$/GPU-giờ` thuần vì card rẻ hơn có thể không chứa vừa mô hình hoặc làm nghẽn băng thông (decode là memory-bound).

**Ext 3 — `cache_is_worth_it()`.** Break-even = phí ghi / (1 − giá đọc) = {c['by_tier']['small']['breakeven_5min']:.2f} lần đọc (TTL 5 phút) hoặc {c['by_tier']['small']['breakeven_1h']:.2f} (TTL 1 giờ). Dữ liệu: tier small ~{c['by_tier']['small']['observed_avg_reads']:.1f} lần đọc/prefix, large ~{c['by_tier']['large']['observed_avg_reads']:.1f} → cache có lợi. Khi tính phí ghi và chỉ bật cache ở nhóm đạt ngưỡng, savings inference là {r2['cache_adjusted_pct']}% (so với {r2['savings_pct']}% headline).

**Ext 4 — Ngân sách reasoning.** Reasoning chiếm {rx['traffic_share']:.1%} request, {rx['usd_share']:.1%} chi phí nhưng {rx['wh_share']:.1%} năng lượng (hệ số ~80×). Giới hạn xuống 5% traffic tiết kiệm ${cap5['usd_saved_daily'] * 30:,.0f}/tháng và {cap5['wh_saved_daily'] * 30 / 1000:,.0f} kWh/tháng; ngưỡng 10% hiện không có tác dụng vì traffic đã dưới 10%. Quy tắc routing: chỉ reasoning cho job offline/batch VÀ (độ phức tạp cao HOẶC confidence model nhỏ < ngưỡng), kèm ngân sách theo project.

**Ext 5 — Lập lịch nhận thức carbon.** Chuyển job gián đoạn được từ us-east-1 sang {r5x['cleanest']} giảm {r5x['saved_g_cleanest'] / 1000:,.0f} kgCO2e ({r5x['saved_pct_cleanest']:.0f}%) cho {r5x['total_kwh']:,.0f} kWh. Rẻ nhất: {r5x['cheapest']}; sạch nhất: {r5x['cleanest']}; cân bằng (carbon $100/tấn): {r5x['balanced']} (điện ${bal['energy_usd']:.0f} và {bal['carbon_g'] / 1000:.0f} kgCO2e). Tiền điện chỉ ~{r5x['home']['energy_usd'] / r5x['gpu_spot_rent']:.0%} tiền thuê GPU nên carbon gần như "miễn phí" để tối ưu; ràng buộc thật là sẵn có GPU, egress dữ liệu và độ trễ (ổn cho training, không ổn cho inference gần người dùng)."""

    return [lie_block, prio_block, ext_block]


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    r5x = ext5_carbon_aware.run(verbose=False)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": r5x["cleanest"],
        "cheapest_region": r5x["cheapest"],
        "balanced_region": r5x["balanced"],
    }

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        unit_economics={"baseline_per_m": r2["baseline_per_m"], "optimized_per_m": r2["optimized_per_m"]},
        extra_sections=_analysis_sections(r1, r2, r3, r5x, levers, baseline))
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"), baseline=baseline)

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
