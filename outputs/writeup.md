# Write-up Lab 25 — GPU FinOps cho NimbusAI

Mọi con số lấy từ `python verify.py`, `python missions/run_all.py` (seed=25, tái lập được) và `outputs/report.md`.

## 1. Baseline vs. Optimized

| | Baseline | Optimized |
|---|---|---|
| Chi phí GPU / tháng | $27,133 | $14,626 |
| Inference `$/1M-token` | $6.488 | $1.126 (−83%) |
| **Tiết kiệm tổng** | | **$12,507 (46%)** |

Baseline = toàn bộ inference chạy model lớn, không cache, không batch (M2) + toàn bộ workload on-demand (M3). Lưu ý phương pháp: hai lever "Right-size" và "Kill idle" được đo trên fleet telemetry (11 GPU), còn baseline lấy từ fleet workloads — hai nguồn này không trùng hoàn toàn, nên 46% là ước lượng tổng hợp theo cách bài lab định nghĩa chứ không phải một hóa đơn duy nhất.

## 2. Phân tích từng đòn bẩy

| Lever | Tiết kiệm/tháng | % baseline |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | 37.0% |
| Inference (cascade/cache/batch) | $1,212 | 4.5% |
| Right-size util-lies | $655 | 2.4% |
| Kill idle GPUs | $600 | 2.2% |

- **Purchasing đóng góp nhiều nhất** vì chi phí training H100 chiếm phần lớn hóa đơn và 5/8 job gián đoạn được → spot có checkpoint; cùng với reserved cho 3 dịch vụ inference 24/7, purchasing giảm 39.1% chi phí workload ($25,667 → $15,627, M3).
- **Inference giảm 83% theo đơn vị nhưng chỉ 4.5% tổng tiền**, vì mới 2,400 request/ngày (~$49/ngày baseline). Lever này sẽ lớn dần theo lưu lượng, trong khi purchasing là chi phí cố định theo hạ tầng.
- Trong 3 đòn bẩy inference, cascade (model nhỏ rẻ ~15×) là lever lớn nhất; batch (−50%) và cache (−90% phần input đã cache) nhân với nhau (batch + 100% cache = 0.05 của giá gốc).

## 3. GPU-Util "lie"

`verify.py` chỉ yêu cầu `gpu-h100-4`, nhưng `flag_util_lies()` thực tế bắt **hai** GPU: `gpu-h100-4` (H100, util 98.2%, MFU 0.194) và `gpu-a10g-1` (A10G, util 96.9%, MFU 0.268).

Cơ chế: GPU-Util chỉ đo tỷ lệ thời gian có kernel đang chạy, không đo FLOPs. Kernel nhỏ, chờ HBM, chờ dataloader hoặc launch overhead vẫn giữ util ~98% khi tensor core gần như rảnh. Các H100 khỏe cùng cụm đạt MFU 0.40–0.43, nên `gpu-h100-4` cho ~47% sản lượng với đúng giá một H100: ~$1,800/tháng trả tiền, ~$900/tháng không sinh giá trị. Right-size theo số đo (Ext 2) cho thấy đổi phần cứng gần như không cứu được (xem bên dưới), nên gốc rễ nằm ở pipeline, không phải loại GPU.

## 4. Phần mở rộng đã làm (cả 5)

| Ext | Việc đã làm | Kết quả đo được | Insight |
|---|---|---|---|
| 1. Tier policy | `evaluate_tiers` / `recommend_tier_v2`: tỷ lệ reclaim theo GPU (3–10%), so sánh reserved 1yr vs 3yr theo vòng đời dịch vụ, tie-break về ít cam kết hơn; ma trận GPU × duty × interruptible | Cùng cách tính trung thực: policy cũ tiết kiệm **26.4%** ($18,893), policy mới **29.8%** ($18,029) | Reserved 3 năm bị "stranded" nếu dịch vụ sống 24 tháng → 1 năm thắng. `job-infer-search` (L4, 18h/ngày) nên on-demand. Headline M3 (39.1%) lạc quan vì chỉ tính reserved theo giờ dùng; thực tế reserved bị tính đủ 24h. |
| 2. Right-size theo MBU | `recommend_rightsize`: GPU rẻ nhất vẫn đủ VRAM ×1.1, băng thông ≤70% peak, compute ≤0.55 MFU; bảng `$/GB-VRAM` | Chỉ `gpu-a10g-0` A10G→L4 (−20%): **$144/tháng** (NVIDIA-only); $936/tháng nếu chấp nhận port sang MI300X | Không chọn theo `$/GPU-giờ`: card rẻ hơn có thể không chứa vừa mô hình (70 GB vào card 24 GB) hoặc nghẽn băng thông. Quy tắc "hạ một bậc" của M5 sẽ đẩy `gpu-a10g-1` xuống L4 với MBU ~0.70 — sát trần, nên không nên áp mù quáng. |
| 3. `cache_is_worth_it` | Break-even = phí ghi / (1 − giá đọc); mô phỏng sự kiện ghi/đọc theo TTL 5 phút trên log token; chỉ credit cache ở nhóm đạt ngưỡng + tính phí ghi | Break-even **0.28** lần đọc (TTL 5p), **1.11** (TTL 1h). Dữ liệu: small ~3.6, large ~0.6 lần đọc/prefix. Savings điều chỉnh **81.0%** (headline 82.6%) | Cache có lợi ở dataset này; nhóm `large` không tag có tỷ lệ đọc dưới ngưỡng. Nếu chuyển sang TTL 1h, nhóm large (0.6 < 1.11) sẽ lỗ. |
| 4. Ngân sách reasoning | Tách `$` và `Wh` cho `is_reasoning`; what-if giới hạn traffic reasoning | 8.4% request → **16.5% chi phí** nhưng **94.0% năng lượng** (~80×). Cap 5%: −$12/tháng, −358 kWh/tháng | Hiện tại traffic reasoning đã <10% nên cap 10% không tiết kiệm gì. 100% reasoning là batch `nightly-eval`, nên `$` nhỏ nhưng carbon lớn → quản trị bằng ngân sách năng lượng, không chỉ ngân sách tiền. Quy tắc routing: chỉ reasoning cho job offline VÀ (độ phức tạp cao HOẶC confidence model nhỏ thấp), giới hạn theo project. |
| 5. Carbon-aware | `region_table` + `ext5_carbon_aware.py` cho 5 job interruptible (2,147 kWh, PUE 1.2) | us-east-1 → europe-north1: **−751 kgCO2e (92%)**. Rẻ nhất: `us-east-wa`; sạch nhất: `europe-north1`; cân bằng (carbon $100/tấn): `us-east-wa` | Tiền điện chỉ ~7% tiền thuê GPU spot nên carbon gần như miễn phí để tối ưu. Vùng "tối ưu" phụ thuộc ưu tiên: tiền → us-east-wa, CO2 → europe-north1. Trade-off: egress dữ liệu, độ trễ, sẵn có GPU — ổn cho training, không ổn cho inference gần người dùng. |

Lưu ý nhỏ: tài liệu (README/Guide) mô tả `europe-north1` là "rẻ và sạch nhất", nhưng theo bảng giá trong code, `us-east-wa` ($0.055/kWh) mới rẻ nhất; `europe-north1` ($0.09) chỉ sạch nhất. Báo cáo tách riêng ba tiêu chí. Các số mẫu trong Guide (ví dụ idle $3,750/tháng) cũng khác kết quả thực tế của dữ liệu seed=25 (idle $600/tháng); báo cáo dùng số thực tế.

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Chuyển toàn bộ job gián đoạn được sang spot có checkpoint** (training LLM/embed, finetune, sandbox, batch-eval) — lever lớn nhất (~$10K/tháng), không cần cam kết dài hạn. Dùng tỷ lệ reclaim theo từng GPU và đo lại sau 1 tháng.
2. **Điều tra pipeline của `gpu-h100-4` và `gpu-a10g-1` trước khi mua hay đổi phần cứng**: bắt đầu bằng dataloader, batch size, kernel fusion; đưa MFU/MBU lên dashboard thay cho GPU-Util và cảnh báo khi util >90% mà MFU <30%.
3. **Gắn tag đầy đủ rồi mới bật chargeback; bật cascade + batch + cache cho inference** với cổng `cache_is_worth_it` và ngân sách reasoning (cả `$` lẫn Wh). Tag coverage hiện 92% (đủ ngưỡng 80%); xử lý 8% request chưa tag, đồng thời giữ job training mới ở vùng sạch/rẻ (`us-east-wa` hoặc `europe-north1`).
