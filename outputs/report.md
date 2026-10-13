# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

**Inference unit cost:** $6.488 -> $1.126 per 1M tokens (**-83%**)

## Savings by lever

| Lever | Savings (USD) | % of baseline |
|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 4.5% |
| Purchasing (spot/reserved) | $10,040 | 37.0% |
| Right-size util-lies | $655 | 2.4% |
| Kill idle GPUs | $600 | 2.2% |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cleanest region: europe-north1
- Cheapest region: us-east-wa
- Best balanced region: us-east-wa

## GPU-Util "nói dối" — cơ chế và tác động tài chính

GPU phát hiện: `gpu-h100-4` (H100, util 98.2%, MFU 0.194), `gpu-a10g-1` (A10G, util 96.9%, MFU 0.268).

`nvidia-smi` GPU-Util chỉ là tỷ lệ thời gian *có ít nhất một kernel đang chạy*, không đo số FLOPs sinh ra. Một kernel nhỏ, bị chờ HBM (memory stall), chờ dữ liệu từ dataloader/I/O, hoặc launch overhead vẫn giữ util ở ~98% trong khi tensor core gần như rảnh. Vì vậy MFU chỉ ~0.2 (so với 0.40–0.43 ở các H100 khỏe cùng cụm): ta trả đủ giờ GPU nhưng nhận ~1/2 sản lượng của một H100 bình thường. Tác động: mỗi H100 bị lie đốt ~$1,800/tháng on-demand cho ~50% công việc hữu ích, tương đương ~$900/tháng lãng phí/GPU. Cách xử lý đúng là sửa nút thắt (dataloader, batch size, kernel fusion), không phải mua thêm GPU — right-sizing theo số đo (Extension 2) cho thấy đổi phần cứng chỉ tiết kiệm được $144/tháng khi giữ NVIDIA.

## Hành động ưu tiên (theo ROI)

1. **Purchasing (spot/reserved)** — $10,040/tháng (37.0% baseline)
2. **Inference (cascade/cache/batch)** — $1,212/tháng (4.5% baseline)
3. **Right-size util-lies** — $655/tháng (2.4% baseline)
4. **Kill idle GPUs** — $600/tháng (2.2% baseline)

Bảng trên xếp theo giá trị tiết kiệm tuyệt đối. Thứ tự thực thi đề xuất (theo rủi ro và thời gian hoàn vốn): (1) chuyển job gián đoạn được sang spot có checkpoint — lever lớn nhất, không phải cam kết; (2) cascade + batch + caching cho inference — giảm 83% $/1M-token, không rủi ro và tăng giá trị theo lưu lượng (hiện chỉ 2,400 request/ngày nên số tiền tuyệt đối còn nhỏ); (3) reserved 1 năm cho dịch vụ chạy 24/7 sau khi đo duty cycle thật; (4) điều tra dataloader/pipeline của các GPU "lie" trước khi right-size; (5) tắt GPU idle.

## Kết quả các phần mở rộng "Your Turn"

**Ext 1 — Chính sách tier theo tỷ lệ gián đoạn & thời hạn cam kết.** Tính lại cả hai chính sách với cùng cách tính trung thực (reserved bị tính đủ 24h×30 ngày dù dùng hay không; spot dùng tỷ lệ reclaim riêng từng GPU): chính sách đơn giản tiết kiệm 26.4% so với on-demand ($18,893), chính sách mới 29.8% ($18,029). (Lưu ý: con số 26.4% thấp hơn headline 39.1% của M3 vì M3 chỉ tính reserved cho giờ sử dụng, còn thực tế reserved bị tính đủ 24h — headline M3 đang lạc quan với dịch vụ chạy <24h/ngày.) Khác biệt giữa hai chính sách đến từ: reserved 1 năm thay vì 3 năm khi vòng đời dịch vụ ước tính 24 tháng (3 năm bị "stranded"), và `job-infer-search` (L4, 18h/ngày) nên giữ on-demand vì reserved không rẻ hơn khi tính đủ giờ.

**Ext 2 — Right-sizing theo MBU/bộ nhớ/compute.** Chọn GPU rẻ nhất vẫn đủ VRAM (×1.1), băng thông (≤70% peak) và compute (≤0.55 MFU). Chỉ `gpu-a10g-0` đổi A10G→L4 (-20%, $144/tháng); tổng $144/tháng với NVIDIA, $936/tháng nếu chấp nhận port sang MI300X. Không chọn theo `$/GPU-giờ` thuần vì card rẻ hơn có thể không chứa vừa mô hình hoặc làm nghẽn băng thông (decode là memory-bound).

**Ext 3 — `cache_is_worth_it()`.** Break-even = phí ghi / (1 − giá đọc) = 0.28 lần đọc (TTL 5 phút) hoặc 1.11 (TTL 1 giờ). Dữ liệu: tier small ~3.6 lần đọc/prefix, large ~0.6 → cache có lợi. Khi tính phí ghi và chỉ bật cache ở nhóm đạt ngưỡng, savings inference là 81.0% (so với 82.6% headline).

**Ext 4 — Ngân sách reasoning.** Reasoning chiếm 8.4% request, 16.5% chi phí nhưng 94.0% năng lượng (hệ số ~80×). Giới hạn xuống 5% traffic tiết kiệm $12/tháng và 358 kWh/tháng; ngưỡng 10% hiện không có tác dụng vì traffic đã dưới 10%. Quy tắc routing: chỉ reasoning cho job offline/batch VÀ (độ phức tạp cao HOẶC confidence model nhỏ < ngưỡng), kèm ngân sách theo project.

**Ext 5 — Lập lịch nhận thức carbon.** Chuyển job gián đoạn được từ us-east-1 sang europe-north1 giảm 751 kgCO2e (92%) cho 2,147 kWh. Rẻ nhất: us-east-wa; sạch nhất: europe-north1; cân bằng (carbon $100/tấn): us-east-wa (điện $118 và 193 kgCO2e). Tiền điện chỉ ~7% tiền thuê GPU nên carbon gần như "miễn phí" để tối ưu; ràng buộc thật là sẵn có GPU, egress dữ liệu và độ trễ (ổn cho training, không ổn cho inference gần người dùng).

_Figures are June-2026 as-of snapshots; re-baseline before acting._