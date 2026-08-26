## Subject: Project A — Audit kết quả và xin góp ý hướng hoàn thiện bài

Kính gửi Thầy,

Em đã hoàn thiện vòng audit và đóng gói Project A về ảnh hưởng của **structural non-reporting** trong dữ liệu giám sát bệnh theo tuần.

**Project làm gì?**

Bài toán xuất phát từ việc bản tin sau tuần 2023w32 chỉ liệt kê 10/16 bệnh có tổng ca toàn tỉnh cao nhất tuần đó. Vì vậy, một bệnh không xuất hiện có thể là không được báo cáo do cơ chế xếp hạng, chứ không đồng nghĩa với zero ca. Project xây dựng panel đầy đủ huyện–tuần–bệnh, tách bốn trạng thái quan sát, rồi định lượng sai lệch khi xử lý mọi ô vắng mặt như zero.

**Method chính**

- Panel `district × week × condition` với observed, reported-but-NR, district/cell-unreported và structurally-unreported.
- Rolling-origin evaluation, horizon 1/2/4 tuần, khóa 20 origins cuối làm test block.
- So sánh zero-fill với reporting-aware trên baseline, Poisson/NB, Random Forest và LightGBM.
- Audit cơ chế presence bằng logistic; audit động bằng discrete-time hazard.
- Bổ sung reporting-propensity train-only làm feature; kiểm tra 5 seed, bootstrap CI và IPW negative control.
- Đánh giá uncertainty bằng empirical residual, quantile regression và CQR.
- Case study trajectory, sensitivity theo quality flags, district heterogeneity, controlled missingness simulation và runtime benchmark.

**Kết quả chính**

- Presence có MNAR-like/right-censoring: beta `2.31`, p `6.1×10⁻²⁰`, AUROC `0.805` chỉ với volume gần nhất và `0.925` khi thêm condition fixed effects.
- Khi đã vắng mặt, duration dự đoán tái xuất hiện tốt hơn volume: AUROC `0.815` so với `0.545`.
- Propensity feature cải thiện MAE ở cả 6/6 nhóm condition × horizon sau khi fit train-only; gain lớn hơn ở nhóm complete và nhỏ nhưng ổn định ở nhóm rotating.
- IPW đơn giản không cải thiện complete và làm xấu rotating; em giữ kết quả âm này như một negative control, không khẳng định IPW nói chung vô dụng.
- Zero-fill làm méo trajectory: Measles `81.0%`, TB `66.3%` số tuần bị censor có relative difference >25%.
- Ở nhóm rotating bị censor 5–8/8 tuần, coverage 80% là `58.3%` với zero-fill và `78.5%` với reporting-aware. Tuy nhiên correlation cấp từng dự báo giữa missing-run và scaled error/interval width gần zero hoặc âm, nên em đã sửa báo cáo để không overclaim tín hiệu đơn điệu.
- CQR đạt coverage gần nominal ở các tổ hợp đủ dữ liệu; 30 tổ hợp thiếu train/calibration được công bố riêng.
- Pipeline chạy nhẹ: dựng panel `6.49 s`, refresh 616 chuỗi `0.82 s`, peak memory `42.4 MB`.

**Kết luận hiện tại**

Missingness ở đây không phải nhiễu có thể bỏ qua. Mô hình hóa observation process bằng một propensity train-only có ích cho dự báo trong protocol rolling-origin, nhưng kết quả nên được gọi là **mechanism-informed/predictive**, không phải causal effect của incidence thật. CQR và trajectory analysis củng cố thông điệp về evaluation validity; IPW cho thấy correction phải khớp với estimand đánh giá.

**Em xin Thầy góp ý 4 điểm**

1. Thầy có đồng ý giữ luận điểm trung tâm là “missing does not mean zero” và đặt novelty ở observation-process/evaluation validity thay vì cạnh tranh thuật toán dự báo không ạ?
2. Thầy muốn dùng cụm “MNAR-like/right-censoring” hay giảm mức khẳng định xuống “non-random structural reporting” để an toàn hơn về mặt phương pháp?
3. Có nên giữ IPW negative control và RQ4 coverage theo nhóm trong bản chính, hay chuyển một phần sang supplementary để bài gọn hơn?
4. Thầy muốn em ưu tiên format theo ICS 2026 hay mở rộng thành bản journal-style trước?

Em gửi kèm bản báo cáo đã audit và thư mục code/kết quả tái lập. Rất mong Thầy góp ý về framing, mức độ khẳng định causal và lựa chọn thí nghiệm giữ lại trong bản nộp.

Em cảm ơn Thầy.

Trân trọng,

[Tên sinh viên]
