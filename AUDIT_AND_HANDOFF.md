# Project A — Audit, xử lý và bàn giao

Ngày audit: 2026-08-26

## Các vấn đề phát hiện và đã xử lý

1. `manuscript/results_data.json` có nhiều placeholder `null` dù kết quả local đã tồn tại. Đã sửa `project_a_consolidate_results.py`, rebuild JSON và xác nhận không còn mục pending.
2. Mô phỏng có cột `delta` legacy không phải zero-minus-aware. Đã loại cột cũ và đổi tên `delta_zero_minus_aware` thành `delta` trong nguồn tổng hợp.
3. Diễn giải RQ4 nói correlation dương, nhưng số liệu thực là `−0.0198` (error), `−0.0633` (width80), `−0.0388` (width90). Đã sửa báo cáo: chỉ giữ kết luận coverage theo nhóm, không claim monotonic signal.
4. CQR bị mô tả như guarantee chung. Đã sửa thành coverage gần nominal dưới giả định exchangeability trên các tổ hợp đủ dữ liệu; công bố 30 tổ hợp bị skip.
5. Propensity bị gọi là “causal feature”. Đã hạ thành `mechanism-informed/predictive train-only`; báo cáo không còn tuyên bố causal effect của incidence thật.
6. Mô phỏng missingness bị gọi là bằng chứng nhân quả trực tiếp. Đã sửa thành controlled simulation evidence trong protocol cụ thể.
7. Thêm mục Audit và tái lập vào DOCX, ghi rõ train-only cutoff `t < 144`, số fits/skips/failures và giới hạn visual QA.

## Kết quả được phép dùng trong email/paper

- Panel: 115,456 ô đầy đủ; 43,384 structurally unreported; 12,248 district/cell unreported; 291 reported-but-NR.
- MNAR-like audit: beta 2.31, 95% CI [1.81, 2.81], p 6.1e-20; AUROC 0.805 volume-only, 0.925 với condition FE.
- Hazard: 623 person-period rows, 37 reappearance events; duration-only AUROC 0.815, volume-only 0.545.
- Propensity feature: ensemble gain dương ở 6/6 nhóm; rotating gain nhỏ nhưng CI loại 0.
- IPW: không giúp complete và làm xấu rotating; chỉ kết luận cho IPW variant/protocol đã kiểm tra.
- Uncertainty: empirical residual under-covered; quantile/CQR gần nominal ở tổ hợp đủ dữ liệu.
- Trajectory distortion: Measles 81.0%, TB 66.3% censored weeks có relative difference >25%.
- Simulation: zero − aware = +0.76, +0.56, +1.71 ở 10%, 20%, 40% missingness.
- Runtime: 6.49 s build panel; 0.82 s refresh 616 series; 42.4 MB peak.

## Còn cần làm trước khi nộp

- Render DOCX bằng LibreOffice/Word và xem từng trang ở 100%.
- Xác nhận DOI/tên tác giả nguồn dữ liệu và format theo template ICS.
- Nếu muốn claim “causal”, cần một thiết kế estimand rõ hơn; bản hiện tại nên dùng “mechanism-informed” hoặc “predictive”.
- Nếu muốn biến RQ4 thành novelty mạnh hơn, cần thêm calibration/coverage theo missing-run với sample size lớn hơn và kiểm định ngoài thời gian.
