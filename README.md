# Project A — Complete Reproducibility Package

## Mục tiêu

Project A kiểm tra một vấn đề dữ liệu trong giám sát bệnh theo tuần: một bệnh không xuất hiện trong bản tin có thể là **structurally unreported**, không phải zero ca. Gói này chứa code tác giả, dữ liệu đầu vào đã kiểm tra, kết quả tóm tắt, hình và bản báo cáo đã audit.

## Cấu trúc

- `src/`: pipeline dựng panel, baseline/model, novelty experiments, consolidation, figures/tables.
- `data/project_a_data/`: dữ liệu đầu vào và README nguồn.
- `results/`: manifests và CSV/JSON nhỏ dùng để audit; prediction dumps lớn không được nhân bản vào gói này.
- `figures/`: 8 hình được nhúng trong báo cáo.
- `manuscript/ProjectA_BaoCao_Draft_Audited.docx`: bản DOCX sau audit.

## Thứ tự chạy an toàn

1. Cài Python 3.12 và các package: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `lightgbm`, `matplotlib`, `python-docx`.
2. Đặt working directory là thư mục gốc project.
3. Chạy các pipeline cục bộ trên bản sao dữ liệu; không chạy Modal trước khi smoke-test thành công.
4. Sau khi mọi output không lỗi, chạy `src/project_a_consolidate_results.py` để tạo `manuscript/results_data.json`.
5. Chạy builder DOCX trong repo gốc (`manuscript/build_report.js`) nếu cần sinh lại báo cáo.

## Quy tắc đánh giá

- Test block cuối gồm 20 rolling origins và không được dùng để fit propensity, chọn model hoặc calibrate interval.
- Báo cáo lỗi chỉ trên target được quan sát.
- CQR có 30 tổ hợp condition–horizon–mode bị skip vì thiếu train/calibration; không được gộp chúng vào denominator.
- Propensity được gọi là **mechanism-informed/predictive**, không phải ước lượng causal effect của incidence thật.
- Mô phỏng missingness là thí nghiệm có kiểm soát trên protocol này, không phải bằng chứng nhân quả về gánh nặng dịch bệnh.

## Kết quả đã audit

- MNAR-like presence model: beta `2.31`, p `6.1e-20`, AUROC volume-only `0.805`, có condition FE `0.925`.
- Hazard: duration-only AUROC `0.815`, volume-only `0.545`, 37 events/623 person-period rows.
- Propensity feature: ensemble CI loại 0 ở 6/6 nhóm condition × horizon; 330 fits, 30 skips, 0 failures.
- IPW đơn giản: không cải thiện complete; làm xấu rotating.
- RQ4: rotating + censor 5–8/8 tuần có coverage 80% `58.3%` zero-fill và `78.5%` reporting-aware; correlation cấp dự báo với missing-run gần zero hoặc âm, nên không claim tín hiệu đơn điệu.
- CQR: coverage gần nominal ở tổ hợp đủ dữ liệu; 30 tổ hợp bị skip.
- Benchmark: build panel `6.49 s`, refresh `0.82 s`, peak memory `42.4 MB`.

## Ghi chú kiểm tra DOCX

DOCX đã được kiểm tra cấu trúc bằng python-docx, heading audit và image audit. Visual render bằng LibreOffice/soffice chưa hoàn tất vì môi trường hiện tại không có `soffice`; cần chạy lại render QA trước khi gửi bản in cuối.
