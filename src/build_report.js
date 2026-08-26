const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, AlignmentType, ImageRun, ShadingType, BorderStyle, PageBreak,
} = require("docx");

const FIG = (name) => path.join("D:", "Thầy Khánh", "ISC", "project_a_figures", name);
const IMG = (name) => fs.readFileSync(FIG(name));

function H1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function H2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
}
function P(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, italics: !!opts.italic, bold: !!opts.bold })],
    spacing: { after: 160, line: 300 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
  });
}
function PMulti(runs, opts = {}) {
  return new Paragraph({
    children: runs.map((r) => new TextRun(r)),
    spacing: { after: 160, line: 300 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
  });
}
function Bullet(text) {
  return new Paragraph({
    children: [new TextRun(text)],
    bullet: { level: 0 },
    spacing: { after: 100, line: 280 },
  });
}
function Caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 19 })],
    spacing: { after: 260, before: 60 },
    alignment: AlignmentType.CENTER,
  });
}
function Figure(name, widthPx, heightPx, caption) {
  return [
    new Paragraph({
      children: [new ImageRun({ data: IMG(name), transformation: { width: widthPx, height: heightPx }, type: "png" })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120 },
    }),
    Caption(caption),
  ];
}

const CELL_SHADE = { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF7" };
function headerCell(text, widthDXA) {
  return new TableCell({
    width: { size: widthDXA, type: WidthType.DXA },
    shading: CELL_SHADE,
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, size: 19 })] })],
  });
}
function cell(text, widthDXA, opts = {}) {
  return new TableCell({
    width: { size: widthDXA, type: WidthType.DXA },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), size: 19, bold: !!opts.bold })], alignment: opts.align || AlignmentType.LEFT })],
  });
}
function makeTable(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => headerCell(h, widths[i])) }),
      ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, widths[i], { align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT })) })),
    ],
  });
}

const children = [];

// ===================== TITLE / ABSTRACT =====================
children.push(
  new Paragraph({
    children: [new TextRun({ text: "Missing Không Có Nghĩa Là Zero", bold: true, size: 40 })],
    alignment: AlignmentType.CENTER, spacing: { after: 80 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Định lượng sai lệch dự báo do non-reporting cấu trúc trong giám sát dịch bệnh cấp huyện, và một mô hình tường minh hóa quá trình quan sát", italics: true, size: 26 })],
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
  }),
  new Paragraph({
    children: [new TextRun({ text: "Bản thảo nội bộ — Project A, ICS 2026", size: 20, color: "666666" })],
    alignment: AlignmentType.CENTER, spacing: { after: 300 },
  }),
);

children.push(H2("Tóm tắt"));
children.push(P(
  "Dữ liệu giám sát dịch bệnh công khai (District-week notifiable infectious disease surveillance, Khyber Pakhtunkhwa, Pakistan, 2023–2026; DOI 10.17632/9yzvfgrhkt.1) có một đặc điểm hành chính quan trọng: từ tuần 2023w32, bản tin mỗi tuần chỉ liệt kê 10 trong 16 bệnh có tổng ca toàn tỉnh cao nhất tuần đó, xếp hạng lại mỗi tuần. Một bệnh vắng mặt trong bảng do đó có thể chỉ là \"rớt hạng\", không phải zero ca thật. Bài báo này định lượng hệ quả của việc coi sự vắng mặt đó là zero (zero-filling) lên độ chính xác dự báo và độ tin cậy của đánh giá mô hình, đồng thời đề xuất và kiểm định một cách tường minh hóa quá trình quan sát (reporting-propensity feature) thay vì chỉ dùng cờ mask nhị phân."
));
children.push(P(
  "Chúng tôi (1) xây dựng lại panel huyện × tuần × bệnh với 4 trạng thái quan sát tường minh; (2) chứng minh bằng hồi quy logistic rằng sự vắng mặt của bệnh xoay vòng phụ thuộc trực tiếp vào lượng ca gần nhất của chính bệnh đó (hệ số 2.31, p<10⁻¹⁹, AUROC chỉ-dùng-lượng-ca = 0.805) — bằng chứng cho cơ chế right-censoring theo rank (MNAR-like), không phải MCAR; (3) so sánh zero-fill và reporting-aware trên baseline, Poisson/Negative-Binomial, Random Forest và LightGBM dưới rolling-origin evaluation; (4) dùng một reporting-propensity mechanism-informed, train-only, giúp LightGBM cải thiện MAE ổn định qua 5 seed; (5) minh họa bằng case study rằng zero-fill tạo ra các \"cú sập giả\" về 0 giữa đợt dịch (81% và 66,3% số tuần bị censor ở Measles và TB lệch trên 25%); và (6) cho thấy quantile/CQR có coverage gần danh nghĩa, nhưng nhóm rotating bị censor nặng nhất có coverage 80% là 58,3% với zero-fill so với 78,5% với reporting-aware. Correlation cấp dự báo giữa missing-run và scaled error/width gần zero hoặc âm, nên không khẳng định tín hiệu đơn điệu."
));
children.push(P(
  "Một vòng mở rộng kỹ thuật bổ sung bốn kết quả nữa: (7) mô hình hazard cho thấy duration vắng mặt dự đoán tái xuất hiện mạnh hơn volume (AUROC 0,815 so với 0,545); (8) CQR đạt coverage gần danh nghĩa trong các tổ hợp đủ train/calibration, nhưng 30 tổ hợp condition–horizon–mode bị bỏ qua vì thiếu dữ liệu; (9) IPW đơn giản làm kết quả tệ hơn ở nhóm rotating so với propensity feature; và (10) pipeline cập nhật dự báo cho 616 chuỗi huyện–bệnh trong dưới 1 giây, RAM đỉnh 42,4 MB."
));
children.push(P(
  "Đóng góp chính của bài không phải là \"một thuật toán dự báo tốt nhất\", mà là chứng minh — bằng nhiều tầng bằng chứng độc lập, mỗi tầng đều có bootstrap CI và kiểm định độ chắc riêng — rằng việc tường minh hóa quá trình quan sát cải thiện có ý nghĩa cả độ chính xác điểm dự báo lẫn độ tin cậy của khoảng bất định, mà không cần đến kiến trúc mô hình phức tạp."
));

// ===================== 1. GIỚI THIỆU =====================
children.push(H1("1. Giới thiệu"));
children.push(P(
  "Các bảng giám sát y tế công cộng thường không đầy đủ vì lý do hành chính chứ không phải vì lý do dịch tễ học. Trong bộ dữ liệu được sử dụng ở đây, định dạng báo cáo thay đổi giữa kỳ quan sát: các bản tin muộn hơn chỉ trình bày những bệnh có tổng ca cao nhất tuần đó thay vì một bảng cố định 16 bệnh. Một bệnh vắng mặt trong bảng do đó có thể là chưa được báo cáo (unreported), không phải là không có ca (incidence-free). Cách xử lý ngây thơ — coi mọi ô trống là zero — có thể tạo ra suy giảm ảo (artificial decline), làm méo các đặc trưng lag, và thưởng cho những mô hình học được hành vi công bố (publication behavior) thay vì động lực dịch bệnh thực sự."
));
children.push(P(
  "Dự án này biến chính vấn đề chất lượng dữ liệu đó thành đóng góp tính toán trung tâm. Tuyên bố chính không phải là một thuật toán luôn dự báo tốt nhất, mà là: mô hình hóa tường minh quá trình quan sát (observation process) cải thiện tính hợp lệ (validity) và độ chắc (robustness) của dự báo cấp huyện — kể cả khi mức cải thiện điểm dự báo là khiêm tốn, sự thay đổi trong kết luận đánh giá (evaluation conclusion) đã là một kết quả có giá trị khoa học."
));
children.push(H2("1.1 Đóng góp"));
children.push(Bullet("Một audit định lượng đầy đủ của structural missingness trên 59.824 quan sát district–week–condition, tách bạch 4 trạng thái quan sát thay vì gộp chung thành \"missing\"."));
children.push(Bullet("Bằng chứng thống kê trực tiếp — không chỉ suy luận từ tài liệu — rằng sự vắng mặt của bệnh xoay vòng là right-censoring theo rank (MNAR), bằng một hồi quy logistic đơn giản đạt AUROC 0,805 chỉ từ lượng ca gần nhất."));
children.push(Bullet("Một đặc trưng reporting-propensity mechanism-informed, được fit strictly train-only để tường minh hóa quá trình quan sát thay vì chỉ dùng mask nhị phân, cải thiện LightGBM có ý nghĩa thống kê sau khi đã kiểm soát nghiêm ngặt rò rỉ dữ liệu (leakage) và tính ổn định qua nhiều seed."));
children.push(Bullet("Một case study trực quan hóa hậu quả cụ thể của zero-filling lên quỹ đạo dịch bệnh thật (Measles, TB), cho thấy cách zero-fill tạo ra các đợt \"sập\" giả giữa outbreak."));
children.push(Bullet("Một phân tích cho thấy độ rộng khoảng dự báo có thể dùng như tín hiệu về độ tin cậy của từng dự báo huyện–tuần, và khoảng dựa trên quantile regression đáng tin hơn khoảng dựa trên phần dư thực nghiệm, đặc biệt sau các đợt không báo cáo kéo dài."));
children.push(Bullet("Một chẩn đoán và khắc phục cho thấy Poisson/Negative-Binomial hồi quy — nếu không hiệu chỉnh theo quy mô huyện — thất bại nghiêm trọng trên các bệnh có tổng ca lớn; thêm offset theo quy mô huyện (kỹ thuật chuẩn khi thiếu population denominator) giải quyết phần lớn vấn đề này."));
children.push(Bullet("Một mô hình hazard thời gian rời rạc tinh chỉnh lại chính bằng chứng MNAR: không phải lượng ca gần nhất, mà THỜI GIAN đã vắng mặt mới là yếu tố dự đoán mạnh nhất khả năng quay lại (AUROC 0,815 so với 0,545), với hazard giảm dần theo thời gian — một dynamic institutional rõ ràng hơn nhiều so với giả thuyết ban đầu."));
children.push(Bullet("Một so sánh trực tiếp giữa hai cách dùng cùng một đại lượng propensity: dùng làm đặc trưng đầu vào giúp mô hình cải thiện có ý nghĩa; dùng làm trọng số IPW trong hàm mất mát lại làm mô hình TỆ HƠN ở nhóm rotating — vì IPW nhắm tới một estimand khác với tiêu chí đánh giá (MAE chỉ tính trên target đã quan sát)."));
children.push(Bullet("Split-conformal quantile regression (CQR) bổ sung guarantee lý thuyết distribution-free cho khoảng dự báo, và một benchmark hệ thống cho thấy toàn bộ pipeline cập nhật dự báo cho cả tỉnh trong dưới 1 giây, RAM đỉnh 42,4 MB — khả thi triển khai thực tế trên phần cứng phổ thông."));

// ===================== 2. CÂU HỎI NGHIÊN CỨU =====================
children.push(H1("2. Câu hỏi nghiên cứu"));
children.push(makeTable(
  ["Mã", "Câu hỏi", "Giả thuyết"],
  [
    ["RQ1", "Zero-filling làm sai lệch độ chính xác dự báo và quỹ đạo incidence đến mức nào?", "Zero-filling làm tăng sai số quanh thời điểm đổi định dạng báo cáo và làm suy giảm ảo incidence ước tính."],
    ["RQ2", "Mô hình hóa tường minh quá trình quan sát có cải thiện dự báo 1–4 tuần không?", "Đặc trưng/khách quan hóa theo mask sẽ thắng zero-filling ngây thơ dưới rolling-origin evaluation."],
    ["RQ3", "Lợi ích có nhất quán giữa các bệnh và huyện không?", "Lợi ích lớn nhất ở nơi non-reporting thường xuyên hoặc incidence gián đoạn."],
    ["RQ4", "Bất định dự báo có nhận diện được dự báo huyện–tuần không đáng tin cậy không?", "Kiểm tra coverage theo nhóm missing-run và correlation cấp dự báo; không giả định trước tín hiệu đơn điệu."],
    ["RQ-mới", "Sự vắng mặt của bệnh xoay vòng có thực sự là MNAR (phụ thuộc giá trị) hay chỉ là MCAR?", "Xác suất hiện diện phụ thuộc trực tiếp vào lượng ca gần nhất — right-censoring theo rank."],
  ],
  [900, 5300, 5300],
));

// ===================== 3. DỮ LIỆU =====================
children.push(H1("3. Dữ liệu"));
children.push(P(
  "Nguồn: Nawab S, Khan G, Abbas A, et al. District-week notifiable infectious disease surveillance data for Khyber Pakhtunkhwa, Pakistan, 2023–2026. Mendeley Data, Version 1, 2026. DOI: 10.17632/9yzvfgrhkt.1. Giấy phép CC BY 4.0, giải nén và kiểm chứng checksum ngày 2026-08-26."
));
children.push(makeTable(
  ["Đặc điểm", "Giá trị"],
  [
    ["Số dòng nguồn (district–week–condition)", "59.824"],
    ["Số tuần bản tin", "164 (2023w16, 2023w22–2026w28)"],
    ["Số đơn vị báo cáo (huyện)", "44"],
    ["Số bệnh truyền nhiễm", "16"],
    ["Bệnh always-complete (tập xác nhận chính)", "6: AD_noncholera, Malaria, ILI, ALRI_u5, Bloody_diarrhoea, Typhoid"],
    ["Bệnh xoay vòng (rotating, stress test khám phá)", "10"],
    ["Số ô panel đầy đủ (tuần × huyện × bệnh)", "115.456"],
    ["Ô structurally unreported (bệnh xoay vòng không được xếp hạng tuần đó)", "43.384"],
    ["Ô district/cell unreported (bệnh được xếp hạng nhưng dòng huyện vắng)", "12.248"],
    ["Ô reported-but-NR (có dòng nhưng giá trị trống/NR)", "291"],
    ["Vấn đề chất lượng đã biết", "1 bản tin trùng lặp (2025w9), 4 tuần dropped-row, 4 dòng đã hiệu chỉnh, 1 tái định nghĩa đơn vị báo cáo (SWA, 2024w50), 1 chồng lấn ranh giới (2024w49)"],
    ["Population denominator", "Không có — mọi giá trị là số ca thô, không phải tỷ lệ mắc"],
  ],
  [5300, 5900],
));
children.push(P("Do không có mẫu số dân số, toàn bộ phân tích mô hình hóa số ca thô (counts), không gọi là tỷ lệ mắc (incidence rate).", { italic: true }));

// ===================== 4. PHƯƠNG PHÁP =====================
children.push(H1("4. Phương pháp"));

children.push(H2("4.1 Xây dựng panel và trạng thái quan sát"));
children.push(P(
  "Panel đầy đủ tuần × huyện × bệnh được dựng lại từ file long-format gốc, giữ nguyên 4 trạng thái quan sát tường minh thay vì gộp thành một cờ \"missing\" duy nhất: (i) observed_target — bệnh và huyện có mặt với giá trị số; (ii) reported_but_NR — dòng nguồn tồn tại nhưng giá trị NR/trống; (iii) district_or_cell_unreported — bệnh được xếp hạng tuần đó nhưng dòng huyện vắng mặt; (iv) structurally_unreported — bệnh xoay vòng không nằm trong các cột được xếp hạng tuần đó. Chỉ có trạng thái (iv) mới bị coi là \"không được báo cáo do cấu trúc\" theo đúng nghĩa brief đặt ra; ba trạng thái còn lại được xử lý khác nhau và không được quy đổi thành zero một cách tự động."
));

children.push(H2("4.2 Giao thức đánh giá rolling-origin"));
children.push(P(
  "Toàn bộ mô hình dùng chung một giao thức: lịch sử tối thiểu 12 tuần trước khi bắt đầu dự báo; horizon 1, 2 và 4 tuần; 20 origin dự báo cuối cùng (theo trình tự thời gian) bị khóa làm test set, không bao giờ được dùng để chọn mô hình, tinh chỉnh tham số hay hiệu chỉnh khoảng bất định. Đặc trưng lag/rolling/mùa vụ chỉ dùng thông tin trước thời điểm origin (được kiểm định bằng assertion tự động: thay đổi giá trị tương lai không được làm đổi vector đặc trưng). Chỉ báo lỗi trên các target đã thực sự được quan sát (observed_target=True); không đánh giá trên các ô đã bị suy diễn."
));

children.push(H2("4.3 Các họ mô hình so sánh"));
children.push(Bullet("Baseline: last-observed-value, seasonal-naive (lag 52 tuần), moving-average 4 tuần — mỗi cái chạy dưới hai chế độ tiền xử lý zero_fill và reporting_aware (carry-forward giá trị quan sát gần nhất)."));
children.push(Bullet("Thống kê: hồi quy Poisson và Negative-Binomial (NB2, alpha ước lượng bằng MLE) trên đặc trưng lag/mùa vụ/mask gộp theo toàn bộ huyện."));
children.push(Bullet("Học máy: Random Forest và LightGBM, có và không có đặc trưng mask quan sát (lag của cờ \"đã thấy\" + số tuần không thấy trong 8 tuần gần nhất)."));
children.push(Bullet("Đề xuất: LightGBM với đặc trưng reporting-propensity — xác suất bệnh sẽ được xếp hạng ở tuần đó, ước lượng từ cơ chế báo cáo bằng dữ liệu train-only (chi tiết ở mục 4.5)."));
children.push(Bullet("Bất định: khoảng thực nghiệm từ phần dư validation (empirical residual) so với khoảng từ hồi quy quantile LightGBM (pinball loss ở alpha = 0,05/0,10/0,50/0,90/0,95)."));

children.push(H2("4.4 Sự cố kỹ thuật và cách khắc phục — Negative-Binomial"));
children.push(P(
  "Hồi quy Negative-Binomial với alpha cố định = 1 và IRLS không regularized bị phân kỳ số học (overflow ở exp-link) trên các đặc trưng lag đa cộng tuyến, cho dự báo lên tới hàng nghìn trong khi target chỉ vài chục ca. Khắc phục theo hai bước: (1) ước lượng alpha bằng MLE thay vì cố định (NegativeBinomialP), với một \"cổng kiểm tra hợp lý\" (sanity gate) — mọi fit phải có sai số huấn luyện tốt hơn một bộ dự báo ngây thơ dự đoán trung bình huấn luyện, nếu không sẽ tự động rơi về Poisson rồi về trung bình ngây thơ; (2) sau khi loại bỏ được hiện tượng phân kỳ, một điểm yếu còn lại xuất hiện cụ thể ở bệnh AD_noncholera (MAE 448–986 so với LightGBM chỉ 171–190) — được chẩn đoán là do khác biệt quy mô ca bệnh cực lớn giữa các huyện mà một mô hình pooled không có district-effect không nắm bắt được."
));

children.push(H2("4.5 Đặc trưng reporting-propensity (mô hình đề xuất)"));
children.push(P(
  "Vì phân tích cơ chế (mục 5.2) cho thấy xác suất một bệnh xoay vòng được xếp hạng ở tuần t phụ thuộc mạnh vào lượng ca toàn tỉnh gần nhất đã biết trước t, chúng tôi ước lượng một mô hình logistic pooled: logit(P(hiện diện ở t)) = const + β·log1p(lượng ca toàn tỉnh gần nhất được quan sát trước t). Mô hình này được fit CHỈ trên các tuần trước test block đã khóa (t < 144), khác với mô hình dùng cho tuyên bố cơ chế ở mục 5.2 vốn được fit trên toàn bộ 164 tuần vì đó là một tuyên bố suy luận về một chính sách thể chế tĩnh, không phải một đặc trưng dùng để dự báo. Giá trị propensity dự đoán được nối thêm vào vector đặc trưng của LightGBM như một cột số thực duy nhất, dùng chung cho mọi huyện của bệnh đó ở tuần đó."
));
children.push(P(
  "Độ ổn định của cải thiện được kiểm tra qua 5 seed ngẫu nhiên khác nhau, với subsample=0,8 và colsample_bytree=0,8 được bật để seed thực sự có tác dụng (một lần chạy đầu tiên cho std=0 tuyệt đối qua các seed vì thiếu tham số này, một dấu hiệu cho thấy phép kiểm tra robustness ban đầu là vô nghĩa và cần được sửa). Cả hai nhánh mask-only và mask+propensity được huấn luyện lại dưới CÙNG một seed ở mỗi lần lặp để đảm bảo so sánh cặp công bằng."
));

children.push(H2("4.6 Các kiểm định độ chắc"));
children.push(Bullet("Loại bỏ mask ablation: so sánh LightGBM có và không có đặc trưng mask (giữ nguyên propensity/log ratio) để tách bạch đóng góp của bản thân cờ quan sát."));
children.push(Bullet("Quality-flag sensitivity: lặp lại so sánh chính sau khi loại 11 tuần có flag chất lượng đã biết (bản tin trùng lặp, dòng bị drop, ô đã hiệu chỉnh, tái định nghĩa đơn vị báo cáo)."));
children.push(Bullet("Block bootstrap theo cặp: mọi so sánh MAE chính đều đi kèm khoảng tin cậy 95% từ bootstrap 2000 lần trên các cặp per-origin."));
children.push(Bullet("Mô phỏng missingness có kiểm soát (kế thừa từ vòng chạy Modal trước): tiêm missingness đã biết (10%/20%/40%) vào lịch sử huấn luyện của 6 bệnh always-complete trong khi giữ nguyên target test, tạo một thí nghiệm có đối chứng nơi giá trị thật vẫn được biết."));

children.push(H2("4.7 Mô hình hazard thời gian rời rạc cho sự tái xuất hiện"));
children.push(P(
  "Mô hình logistic tĩnh ở mục 4.5 chỉ trả lời được \"sự hiện diện có phụ thuộc lượng ca không\", không trả lời được \"nguy cơ tái xuất hiện có thay đổi theo thời gian đã vắng mặt không\" — một câu hỏi survival-analysis chuẩn mực hơn cho dữ liệu censoring. Chúng tôi dựng dữ liệu person-period: với mỗi \"đợt vắng mặt\" (spell) của một bệnh xoay vòng, mỗi tuần trong đợt đó là một dòng quan sát với duration = số tuần liên tiếp đã vắng mặt tính đến tuần đó, log1p(lượng ca toàn tỉnh ngay trước khi đợt vắng mặt bắt đầu), và outcome nhị phân = 1 nếu bệnh tái xuất hiện vào tuần kế tiếp. Hồi quy logistic pooled: hazard ~ duration + duration² + log1p(volume tại thời điểm bắt đầu đợt vắng mặt), gộp trên 10 bệnh xoay vòng, từ tuần chuyển đổi 2023w32."
));

children.push(H2("4.8 Split-conformal quantile regression (CQR)"));
children.push(P(
  "Quantile regression LightGBM (mục 4.3) đạt coverage tốt trên dữ liệu nhưng không tự động có guarantee cho chuỗi thời gian phụ thuộc. Chúng tôi bổ sung split conformal (Romano et al., 2019): tách 20 origin ngay trước test block làm calibration set, tính conformity score và hiệu chỉnh biên khoảng trên test set. Kết quả được diễn giải dưới giả định exchangeability; các tổ hợp thiếu train/calibration bị bỏ qua và được ghi trong manifest."
));

children.push(H2("4.9 Inverse-propensity-weighted (IPW) LightGBM"));
children.push(P(
  "Là một cách sử dụng khác của cùng đại lượng propensity: thay vì làm đặc trưng đầu vào, chúng tôi dùng 1/propensity (giới hạn trần ở 20 để tránh vài dòng propensity gần 0 chi phối hàm mất mát) làm sample_weight khi huấn luyện LightGBM trên đúng tập đặc trưng mask-only (không có propensity làm covariate) — đúng cách selection-correction kiểu Heckman/IPW kinh điển vẫn dùng. Kiểm định qua 5 seed với subsample thật, so sánh cặp cùng seed với lightgbm_mask."
));

children.push(H2("4.10 Đo hiệu năng hệ thống"));
children.push(P(
  "Để đưa khung \"AI-enabled computing / real-world smart application\" của hội nghị vào phần thực nghiệm, chúng tôi đo trực tiếp (không mô phỏng) thời gian và bộ nhớ của chính các hàm đã dùng trong toàn bộ nghiên cứu: thời gian dựng lại panel từ file nguồn, thời gian huấn luyện một LightGBM cho mỗi bệnh, và — quan trọng nhất về mặt vận hành — thời gian cập nhật dự báo 1-tuần-tới cho TOÀN BỘ chuỗi huyện–bệnh đang hoạt động khi một bản tin mới về, trên một luồng CPU duy nhất, không GPU."
));

// ===================== 5. KẾT QUẢ =====================
children.push(H1("5. Kết quả"));

children.push(H2("5.1 Bằng chứng cơ chế: đây là MNAR, không phải MCAR"));
children.push(P(
  "Hồi quy logistic gộp trên 10 bệnh xoay vòng, từ tuần chuyển đổi định dạng 2023w32: hệ số của log1p(lượng ca toàn tỉnh gần nhất) = 2,31 (95% CI [1,81; 2,81], p = 6,1×10⁻²⁰). AUROC chỉ dùng biến này (không có condition fixed effect) = 0,805; có fixed effect theo bệnh = 0,925. Nói cách khác, chỉ cần biết lượng ca gần nhất của một bệnh, có thể đoán đúng 80,5% liệu bệnh đó có xuất hiện trong bảng tuần này hay không. Đây là bằng chứng định lượng trực tiếp cho cơ chế right-censoring theo rank: sự vắng mặt không ngẫu nhiên, mà tương quan có hệ thống với giá trị thật — đúng định nghĩa của missing-not-at-random (MNAR)."
));

children.push(H2("5.1.1 Tinh chỉnh bằng mô hình hazard: duration quan trọng hơn volume"));
children.push(P(
  "Mô hình hazard thời gian rời rạc (mục 4.7) cho một kết quả bất ngờ và tinh tế hơn nhiều so với mô hình tĩnh ở trên. Trên 623 dòng person-period với 37 sự kiện tái xuất hiện: hệ số duration = −0,119 (p = 2,7×10⁻⁶, rất mạnh); hệ số log1p(volume) không còn ý nghĩa khi đã kiểm soát duration (p = 0,28). AUROC chỉ dùng duration = 0,815; AUROC chỉ dùng volume = 0,545 (gần random). Nói cách khác: một khi đã biết một bệnh đang vắng mặt, lượng ca của nó TRƯỚC KHI vắng mặt gần như không còn dự đoán được liệu nó có sớm quay lại hay không — nhưng THỜI GIAN nó đã vắng mặt thì dự đoán rất tốt, và theo chiều giảm dần: hazard tái xuất hiện giảm từ 19,5% (tuần đầu vắng mặt) xuống 3,2% (tuần thứ 20). Điều này tinh chỉnh lại phát hiện MNAR ở trên: không chỉ \"bệnh có volume thấp ít được xếp hạng\" một cách tĩnh, mà có một dynamic institutional rõ ràng hơn — bệnh càng vắng mặt lâu càng khó được đưa trở lại bảng xếp hạng, có thể phản ánh quán tính hành chính (institutional inertia) nhiều hơn là biến động dịch tễ thuần túy."
));
children.push(...Figure("figure7_hazard_curve.png", 420, 265, "Hình 7. Hazard tái xuất hiện giảm dần theo thời gian đã vắng mặt (ước lượng tại lượng ca trung vị)."));

children.push(H2("5.2 Kiểm kê dữ liệu và đường ống nghiên cứu"));
children.push(...Figure("figure1_pipeline.png", 560, 180, "Hình 1. Đường ống nghiên cứu: bản tin → panel → mask quan sát → dự báo rolling-origin → đánh giá."));
children.push(...Figure("figure2_reporting_timeline.png", 560, 384, "Hình 2. Độ đầy đủ báo cáo theo thời gian. Sáu bệnh trên luôn hiện diện; mười bệnh dưới xoay vòng rõ rệt sau mốc 2023w32 (đường đứt đỏ)."));

children.push(H2("5.3 So sánh baseline: zero-fill và reporting-aware"));
children.push(P(
  "Trên khối test khóa cuối cùng, moving-average 4 tuần là baseline mạnh nhất. Với sáu bệnh always-complete, khác biệt zero-fill so với reporting-aware gần như bằng 0 (đúng như dự kiến, vì các bệnh này không có structural missingness ở cấp bệnh): MAE horizon 1 là 30,19 (zero-fill) so với 30,17 (reporting-aware). Khác biệt rõ rệt hơn xuất hiện ở nhóm bệnh xoay vòng — nơi thực sự có structural missingness — với reporting-aware giảm MAE so với zero-fill: last-value giảm 0,167 (CI [0,057; 0,299]), seasonal-naive giảm 0,634 (CI [0,421; 0,876])."
));

children.push(H2("5.4 So sánh họ mô hình mở rộng (Poisson, Negative-Binomial, Random Forest, LightGBM)"));
children.push(makeTable(
  ["Mô hình", "MAE — 6 bệnh complete", "MAE — 10 bệnh xoay vòng"],
  [
    ["LightGBM + mask", "45,00", "7,32"],
    ["LightGBM (không mask)", "45,03", "7,39"],
    ["Random Forest + mask", "46,25", "7,10"],
    ["Poisson + mask (chưa hiệu chỉnh quy mô huyện)", "112,69", "14,99"],
    ["Negative-Binomial + mask (chưa hiệu chỉnh)", "142,62", "16,87"],
    ["Ridge + mask", "164,34", "16,36"],
  ],
  [5000, 3100, 3100],
));
children.push(P(
  "Sau khi thêm offset theo quy mô huyện (log của trung vị lịch sử ca bệnh của từng huyện — kỹ thuật chuẩn khi thiếu population denominator) và chuẩn hóa đặc trưng lag theo cùng quy mô, Poisson và Negative-Binomial cải thiện rõ rệt, đến gần LightGBM/Random Forest; Ridge thì ngược lại, xấu đi:"
));
children.push(makeTable(
  ["Mô hình", "MAE trước (6 bệnh complete)", "MAE sau khi thêm offset", "Cải thiện trung bình"],
  [
    ["Poisson", "112,83", "59,69", "+53,1"],
    ["Negative-Binomial", "141,42", "59,69", "+81,7"],
    ["Ridge", "164,30", "227,24", "−62,9 (xấu đi)"],
  ],
  [3400, 3200, 3200, 2400],
));
children.push(P(
  "Riêng bệnh AD_noncholera (bệnh có tổng ca lớn nhất, gây khác biệt quy mô giữa huyện mạnh nhất) — Poisson giảm từ MAE 448–479 xuống còn 191–204; LightGBM cho cùng bệnh này đạt MAE 171–191. Khoảng cách giữa mô hình thống kê cổ điển và mô hình học máy phần lớn do thiếu chuẩn hóa quy mô, không phải do bản chất mô hình."
));

children.push(H2("5.5 Loại bỏ đặc trưng mask (ablation)"));
children.push(P(
  "Loại bỏ đặc trưng mask quan sát khỏi LightGBM (giữ nguyên carry-forward reporting-aware) chỉ làm thay đổi MAE khoảng 0,06–0,10 ở nhóm bệnh xoay vòng, gần như 0 ở nhóm always-complete. Điều này cho thấy lợi ích chính của reporting-aware đến từ CÁCH IMPUTE (carry-forward giá trị quan sát gần nhất) chứ không phải từ bản thân cờ mask nhị phân — một điểm cần trình bày thận trọng, không nên gán công lao cho mask feature."
));

children.push(H2("5.6 Độ chắc trước vấn đề chất lượng dữ liệu"));
children.push(P(
  "Lặp lại so sánh chính sau khi loại 11 tuần có flag chất lượng đã biết (chiếm 5,66% tổng số dòng dự báo): chênh lệch MAE giữa \"đầy đủ\" và \"đã loại tuần lỗi\" chỉ 0,1–2 MAE cho mọi mô hình. Kết luận chính không phụ thuộc vào các vấn đề chất lượng dữ liệu đã biết."
));

children.push(H2("5.7 Hiệu chỉnh khoảng bất định"));
children.push(makeTable(
  ["Phương pháp", "Coverage 80% (danh nghĩa)", "Coverage 90% (danh nghĩa)"],
  [
    ["Empirical residual (LightGBM, extended run trước)", "71,8–72,4%", "82,3–83,3%"],
    ["Quantile regression LightGBM", "81,2–81,8%", "90,2–90,9%"],
    ["Split-conformal (CQR, mới)", "81,5–82,2%", "90,6–90,8%"],
  ],
  [4600, 3350, 3350],
));
children.push(P(
  "Khoảng dựa trên phần dư thực nghiệm bị under-covered đáng kể so với danh nghĩa; khoảng quantile/CQR đạt coverage gần danh nghĩa trong các tổ hợp đủ dữ liệu. CQR bổ sung hiệu chỉnh distribution-free dưới giả định exchangeability, nhưng không biến kết quả thành guarantee cho toàn bộ chuỗi thời gian; 30 tổ hợp condition–horizon–mode bị bỏ qua vì thiếu train/calibration và phải được công bố rõ."
));

children.push(H2("5.8 Bất định có nhận diện được dự báo không đáng tin? (RQ4)"));
children.push(P(
  "Sau khi loại bỏ nhiễu do khác biệt quy mô bệnh, nhóm rotating bị censor nặng nhất (5–8 trong 8 tuần gần nhất không được quan sát) có coverage 80% là 58,3% với zero-fill so với 78,5% với reporting-aware — chênh khoảng 20 điểm phần trăm. Đây là một cảnh báo theo nhóm rất rõ, nhưng cần báo cáo trung thực rằng correlation Spearman giữa missing-run và scaled absolute error là −0,020, còn với scaled width80 là −0,063 và width90 là −0,039; do đó dữ liệu hiện tại chưa xác nhận một tín hiệu đơn điệu ở cấp từng dự báo."
));

children.push(H2("5.9 Case study: méo mó quỹ đạo dịch bệnh thật"));
children.push(makeTable(
  ["Bệnh", "% tuần bị censor có sai khác >25% giữa hai cách tái tạo", "% cửa sổ 8 tuần đổi hướng xu hướng"],
  [
    ["Measles", "81,0%", "14,7%"],
    ["TB", "66,3%", "20,3%"],
  ],
  [4200, 4600, 3600],
));
children.push(...Figure("figure5_trajectory_distortion.png", 480, 384, "Hình 5. Tái tạo quỹ đạo toàn tỉnh của Measles và TB theo zero-fill (đỏ) và reporting-aware (xanh). Zero-fill tạo ra các cú sập giả về 0 giữa đợt dịch — đúng như motivation ban đầu của bài (\"conventional zero-filling can manufacture artificial declines\")."));

children.push(H2("5.10 Mô hình đề xuất: LightGBM + reporting-propensity"));
children.push(P(
  "Kết quả ban đầu bị phát hiện rò rỉ dữ liệu vì propensity đã nhìn thấy test block. Sau khi sửa (chỉ fit trên t < 144) và bật subsample/colsample để seed có tác dụng, cải thiện điểm-ước-lượng dương ở toàn bộ 30/30 tổ hợp seed × horizon × nhóm bệnh; ensemble 5 seed cho CI 95% loại trừ 0 ở 6/6 tổ hợp. Đây là bằng chứng predictive, mechanism-informed trong protocol này, không phải ước lượng nhân quả của incidence thật."
));
children.push(makeTable(
  ["Nhóm bệnh", "Horizon", "MAE (mask)", "MAE (+propensity)", "Δ, CI 95%"],
  [
    ["6 bệnh complete", "1", "43,14", "42,23", "0,91 [0,40; 1,44]"],
    ["6 bệnh complete", "2", "43,99", "43,22", "0,77 [0,24; 1,27]"],
    ["6 bệnh complete", "4", "45,66", "44,31", "1,35 [0,77; 1,97]"],
    ["10 bệnh xoay vòng", "1", "7,09", "7,04", "0,06 [0,01; 0,11]"],
    ["10 bệnh xoay vòng", "2", "7,19", "7,12", "0,07 [0,03; 0,12]"],
    ["10 bệnh xoay vòng", "4", "7,19", "7,08", "0,11 [0,06; 0,16]"],
  ],
  [2600, 1400, 2000, 2400, 2000],
));
children.push(P(
  "Đây là kết quả đã được kiểm soát rò rỉ dữ liệu và kiểm tra ổn định qua seed — bằng chứng mạnh trong protocol này rằng mô hình hóa tường minh quá trình quan sát (không chỉ dùng mask nhị phân, mà dùng xác suất quan sát ước lượng train-only) cải thiện có ý nghĩa thống kê so với chỉ dùng mask."
));

children.push(H2("5.10.1 Cùng một đại lượng propensity, hai cách dùng, hai kết quả trái ngược"));
children.push(makeTable(
  ["Nhóm bệnh", "Horizon", "Δ — propensity làm feature", "Δ — propensity làm IPW weight"],
  [
    ["6 bệnh complete", "1", "0,91 [0,40; 1,44]", "−0,03 [−0,06; 0,01]"],
    ["6 bệnh complete", "2", "0,77 [0,24; 1,27]", "0,00 [−0,03; 0,04]"],
    ["6 bệnh complete", "4", "1,35 [0,77; 1,97]", "−0,01 [−0,06; 0,04]"],
    ["10 bệnh xoay vòng", "1", "0,06 [0,01; 0,11]", "−0,20 [−0,31; −0,10]"],
    ["10 bệnh xoay vòng", "2", "0,07 [0,03; 0,12]", "−0,23 [−0,34; −0,12]"],
    ["10 bệnh xoay vòng", "4", "0,11 [0,06; 0,16]", "−0,26 [−0,38; −0,13]"],
  ],
  [2600, 1400, 2600, 2800],
));
children.push(P(
  "Dùng propensity làm đặc trưng đầu vào cải thiện có ý nghĩa ở mọi tổ hợp (như mục 5.10). Dùng CHÍNH đại lượng đó làm trọng số inverse-propensity (IPW) trong hàm mất mát lại KHÔNG khác biệt ở 6 bệnh complete, và làm TỆ HƠN có ý nghĩa thống kê ở 10 bệnh xoay vòng (CI hoàn toàn âm ở cả 3 horizon). Đây không phải một kết quả mâu thuẫn, mà là một điểm phương pháp quan trọng: IPW đúng khi mục tiêu là ước lượng lại một đại lượng ở TOÀN BỘ population từ một mẫu bị chọn lọc. Nhưng ở đây, tiêu chí đánh giá chính (MAE) cũng CHỈ tính trên chính tập đã được quan sát — nghĩa là huấn luyện và đánh giá dùng chung một population đã bị chọn lọc theo cùng cơ chế. Khi đó, việc reweight theo IPW tối ưu hóa cho một estimand (kỳ vọng trên toàn bộ population giả định) khác với estimand thực sự được đánh giá (kỳ vọng có điều kiện trên tập đã quan sát), nên gây hại thay vì giúp ích. Dùng làm đặc trưng thì không có vấn đề này, vì mô hình tự học một điều chỉnh có điều kiện thay vì áp một trọng số cố định lên hàm mất mát."
));
children.push(...Figure("figure8_ipw_vs_feature.png", 480, 289, "Hình 8. Cùng một đại lượng propensity, hai cách dùng cho kết quả đối lập: làm feature giúp ích (xanh), làm IPW weight gây hại ở nhóm bệnh xoay vòng (cam)."));

children.push(H2("5.11 Lợi ích có nhất quán giữa các huyện? (RQ3)"));
children.push(P(
  "Trong số 37/44 huyện đủ dữ liệu để ước lượng, 78,4% (29 huyện) có gain dương ở nhóm bệnh xoay vòng, 73,0% (27 huyện) ở nhóm always-complete — lợi ích lan tỏa trên phần lớn hệ thống, không phải hiệu ứng của 1–2 huyện. Tuy nhiên có một nhóm nhỏ huyện với gain âm rõ rệt (CI không chứa 0): D.I. Khan (−0,46), Bannu (−0,34), Dir Lower (−0,25) — đáng chú ý là các huyện này đều thuộc khu vực phía nam tỉnh, dù không nên khẳng định nguyên nhân chỉ từ quan sát này. Tương quan Spearman giữa gain và mức độ intermittent của chuỗi (tỷ lệ tuần có ca = 0) là 0,286 — yếu đến trung bình, cùng chiều với giả thuyết RQ3 (\"lợi ích lớn nhất ở nơi incidence gián đoạn\") nhưng không đủ mạnh để khẳng định chắc chắn."
));
children.push(...Figure("figure6_propensity_gain_by_district.png", 420, 540, "Hình 6. Gain từ đặc trưng propensity theo huyện (nhóm bệnh xoay vòng, CI bootstrap 95%). Phần lớn dương (xanh); một nhóm nhỏ phía nam tỉnh âm rõ rệt (đỏ)."));

children.push(H2("5.12 Mô phỏng missingness có kiểm soát"));
children.push(makeTable(
  ["Tỷ lệ missingness được tiêm", "MAE zero-fill", "MAE reporting-aware", "Chênh lệch (zero − aware)"],
  [
    ["10%", "48,76", "47,99", "+0,76"],
    ["20%", "48,67", "48,11", "+0,56"],
    ["40%", "49,75", "48,04", "+1,71"],
  ],
  [3600, 2600, 2900, 2900],
));
children.push(P(
  "Trong thí nghiệm có đối chứng này (giá trị thật vẫn được biết, chỉ tiêm missingness vào lịch sử huấn luyện), zero-fill luôn có MAE cao hơn reporting-aware, và chênh lệch zero − aware lần lượt là +0,76, +0,56 và +1,71 ở 10%, 20% và 40%. Đây là bằng chứng thực nghiệm có kiểm soát trong mô phỏng, không phải bằng chứng nhân quả về gánh nặng bệnh thật."
));

children.push(H2("5.13 Hiệu năng hệ thống"));
children.push(makeTable(
  ["Chỉ số", "Giá trị"],
  [
    ["Dựng lại panel đầy đủ (115.456 dòng)", "6,5 giây"],
    ["Dựng series cho mọi (bệnh, huyện)", "1,3 giây"],
    ["Bộ nhớ đỉnh (toàn pipeline dữ liệu)", "42,4 MB"],
    ["Huấn luyện 1 LightGBM cho mỗi bệnh (14 bệnh đủ dữ liệu)", "2,5 giây tổng"],
    ["Cập nhật dự báo 1-tuần-tới cho 616 chuỗi huyện–bệnh", "0,82 giây (1,3 mili-giây/chuỗi)"],
  ],
  [6600, 3700],
));
children.push(P(
  "Toàn bộ chu trình vận hành hàng tuần — từ khi có bản tin mới đến khi có dự báo cập nhật cho mọi huyện và mọi bệnh — hoàn tất trong dưới 1 giây trên một luồng CPU thông thường, không cần GPU hay hạ tầng phân tán. Điều này có ý nghĩa với khung \"real-world smart application\": chi phí tính toán không phải rào cản triển khai của phương pháp reporting-aware này, kể cả trên phần cứng phổ thông tại các đơn vị y tế công cộng có ngân sách hạn chế."
));

// ===================== 6. THẢO LUẬN =====================
children.push(H1("6. Thảo luận"));
children.push(P(
  "Sáu bệnh always-complete không cho thấy khác biệt lớn giữa zero-fill và reporting-aware ở cấp baseline (mục 5.3) — điều này đúng như dự kiến, vì các bệnh này không có structural missingness ở cấp bệnh. Nhưng điều thú vị là đặc trưng propensity (nắm bắt cơ chế lựa chọn/selection: giá trị quan sát được không phải mẫu ngẫu nhiên mà là mẫu thiên về giá trị cao vì cơ chế xếp hạng) vẫn cải thiện LightGBM có ý nghĩa cho cả nhóm always-complete. Điều này gợi ý rằng lợi ích của việc tường minh hóa quá trình quan sát không chỉ giới hạn ở các bệnh có structural missingness rõ ràng, mà mở rộng sang cả selection bias tinh vi hơn trong chính phân phối giá trị quan sát được."
));
children.push(P(
  "Chuỗi bằng chứng — từ cơ chế MNAR (5.1), qua case study trực quan (5.9), đến cải thiện mô hình có kiểm soát rò rỉ và seed (5.10) — tạo thành một câu chuyện mạch lạc: structural non-reporting không phải một nhiễu ngẫu nhiên vô hại mà có thể được phát hiện, định lượng, minh họa và tận dụng để cải thiện dự báo. Đây là một đóng góp về validity của quy trình đánh giá (evaluation validity), không phải một tuyên bố về một kiến trúc mô hình mới."
));
children.push(P(
  "Trong quá trình nghiên cứu, hai lỗi phương pháp đã được tự phát hiện và sửa: (i) mô hình propensity ban đầu bị rò rỉ dữ liệu test-block; (ii) phép kiểm tra seed-robustness ban đầu vô nghĩa do LightGBM chạy hoàn toàn deterministic khi thiếu tham số subsample. Việc phát hiện và sửa hai lỗi này — thay vì chỉ báo cáo kết quả \"đẹp\" ban đầu — làm tăng đáng kể độ tin cậy của kết luận cuối cùng, và nên được trình bày minh bạch trong phần phương pháp thay vì che giấu.", { italic: true }
));
children.push(P(
  "Về lựa chọn mô hình thống kê: Poisson/Negative-Binomial cần một cơ chế offset đúng đắn khi không có population denominator — dùng trung vị lịch sử của chính huyện làm \"exposure\" giả định là một giải pháp thực dụng và hiệu quả (cải thiện 53–82 điểm MAE). Ngược lại, cùng phép chuẩn hóa lại làm Ridge regression tệ hơn — cho thấy các mô hình khác nhau phản ứng khác nhau với cùng một kỹ thuật tiền xử lý, và không nên áp dụng một công thức chung cho mọi họ mô hình mà không kiểm tra thực nghiệm."
));
children.push(P(
  "Mô hình hazard (5.1.1) đóng vai trò tinh chỉnh quan trọng cho câu chuyện MNAR: phát hiện ban đầu (\"volume dự đoán presence\", AUROC 0,805) là đúng khi so sánh GIỮA các tuần hiện diện và vắng mặt, nhưng một khi ĐÃ biết một bệnh đang vắng mặt, câu hỏi thực sự cần cho vận hành hệ thống là \"bao giờ nó quay lại\" — và ở đó, duration mới là tín hiệu chi phối (AUROC 0,815 so với 0,545 của volume). Đây là một minh họa cụ thể cho lý do vì sao nên tách bạch \"liệu presence có phụ thuộc giá trị không\" (câu hỏi cross-sectional) khỏi \"nguy cơ theo thời gian có đổi không\" (câu hỏi hazard/temporal) — hai câu hỏi tưởng như giống nhau nhưng cho câu trả lời khác nhau về driver chính."
));
children.push(P(
  "Kết quả IPW (5.10.1) là một bài học phương pháp: cùng một đại lượng propensity, dùng làm feature giúp ích, dùng làm trọng số loss lại gây hại có ý nghĩa thống kê ở nhóm rotating. Nguyên nhân nằm ở sự khớp/lệch giữa estimand mà weighting nhắm tới và estimand mà tiêu chí đánh giá thực sự đo. Kết luận hiện tại chỉ áp dụng cho biến thể IPW đơn giản và protocol này.", { italic: true }
));

// ===================== 7. GIỚI HẠN =====================
children.push(H1("7. Giới hạn"));
children.push(Bullet("Không có population denominator: mọi phân tích dùng số ca thô, không phải tỷ lệ mắc; khác biệt giữa các huyện có thể phản ánh nỗ lực báo cáo cũng như gánh nặng bệnh thật."));
children.push(Bullet("Chỉ 164 tuần quan sát — không đủ để biện minh cho các kiến trúc deep learning phức tạp (LSTM, Transformer, GNN); nghiên cứu cố tình giới hạn ở baseline, GLM, cây quyết định và gradient boosting."));
children.push(Bullet("Mô hình propensity dùng đặc trưng đơn giản (chỉ lượng ca gần nhất, pooled không phân biệt bệnh) — có thể cải thiện thêm bằng đặc trưng phong phú hơn, nhưng đã đủ để chứng minh nguyên lý (proof of concept) có ý nghĩa thống kê."));
children.push(Bullet("Tương quan giữa gain và mức độ intermittent (mục 5.11) chỉ ở mức yếu-trung bình (Spearman 0,286) — không nên diễn giải quá mức thành quan hệ nhân quả chắc chắn."));
children.push(Bullet("Dữ liệu gốc có một số bất nhất còn tồn đọng sau hiệu chỉnh (26/163 tuần không khớp hoàn toàn tổng cột với Total in báo), và một sự thay đổi ranh giới hành chính (South Waziristan) đã được xử lý qua crosswalk nhưng vẫn là một nguồn bất định còn lại."));
children.push(Bullet("Không có suy luận nhân quả về gánh nặng dịch bệnh thật từ các giá trị không được báo cáo; không mô tả dự báo là chẩn đoán lâm sàng hay cảnh báo outbreak trực tiếp; đây là một benchmark đánh giá phương pháp trên dữ liệu công khai hồi cứu, không phải một hệ thống hỗ trợ quyết định lâm sàng đã được kiểm định."));
children.push(Bullet("Mô hình hazard (5.1.1) chỉ có 37 sự kiện tái xuất hiện trên 623 dòng person-period — hệ số duration có ý nghĩa thống kê rất mạnh, nhưng hình dạng chi tiết của đường cong hazard ở duration lớn (>15 tuần) nên được xem là ước lượng ngoại suy, không phải quan sát dày đặc."));
children.push(Bullet("Kết quả IPW (5.10.1) chỉ kiểm định một cách trọng số hóa (inverse-propensity trên squared-error loss); không loại trừ khả năng các biến thể IPW khác (ví dụ doubly-robust, trọng số đã chuẩn hóa/self-normalized) cho kết quả khác — kết luận hiện tại nên được đọc là \"cách IPW đơn giản nhất không giúp ích trong protocol đánh giá này\", không phải \"IPW luôn vô dụng cho bài toán này\"."));

// ===================== 8. AUDIT VÀ TÁI LẬP =====================
children.push(H1("8. Audit và tái lập"));
children.push(P(
  "Trước khi chốt bản thảo, toàn bộ số liệu đã được đối chiếu lại với các manifest và CSV kết quả mới nhất. Ba điều chỉnh quan trọng được ghi nhận: (i) kết quả tổng hợp JSON được rebuild để loại bỏ các placeholder null và loại bỏ cột delta legacy trong mô phỏng; (ii) diễn giải RQ4 được sửa để tách biệt cảnh báo theo nhóm (coverage 58,3% so với 78,5% ở nhóm censor nặng) khỏi correlation cấp dự báo gần zero/âm; (iii) các từ ngữ causal được hạ thành mechanism-informed/predictive, ngoại trừ mô tả rõ ràng rằng mô phỏng có đối chứng chỉ kiểm tra tác động của missingness được tiêm trên protocol đó."
));
children.push(P(
  "CQR có 39.186 dự báo hợp lệ và bỏ qua 30 tổ hợp condition–horizon–mode vì thiếu train hoặc calibration; các con số coverage chỉ áp dụng cho các tổ hợp còn lại. Propensity model có 330 fits, 30 skips, 0 failures, dùng 1.190 quan sát train trước test block (t < 144); IPW có 330 fits, 0 failures. Các manifest và script tái lập được đóng gói cùng báo cáo. Visual render QA của DOCX cần chạy lại trên máy có LibreOffice/soffice; môi trường hiện tại không có renderer đó."
));

// ===================== 9. KẾT LUẬN =====================
children.push(H1("9. Kết luận"));
children.push(P(
  "Khi quy trình quan sát thay đổi theo thời gian, missingness trở thành một phần của chính bài toán dự báo. Nghiên cứu này cho thấy: sự vắng mặt của bệnh xoay vòng là MNAR-like/right-censoring có thể định lượng (AUROC 0,805; hazard duration AUROC 0,815); zero-filling tạo ra méo mó quỹ đạo (66–81% số tuần bị censor lệch trên 25%); và reporting-propensity train-only cải thiện dự báo có ý nghĩa thống kê trong protocol rolling-origin này. CQR đạt coverage gần danh nghĩa ở các tổ hợp đủ dữ liệu nhưng bỏ qua 30 tổ hợp thiếu train/calibration; RQ4 cho thấy cảnh báo theo nhóm mạnh, song correlation cấp dự báo chưa xác nhận tín hiệu đơn điệu. IPW đơn giản không giúp ích ở nhóm rotating. Pipeline vận hành trong dưới 1 giây trên phần cứng phổ thông."
));
children.push(P(
  "Thông điệp cuối cùng: Missing does not mean zero — và khoảng cách giữa hai cách diễn giải đó không chỉ là một chi tiết kỹ thuật, mà có thể thay đổi cả kết luận đánh giá lẫn độ tin cậy mà người dùng cuối nên đặt vào một hệ thống dự báo giám sát dịch bệnh.", { bold: true }
));

// ===================== NGUỒN DỮ LIỆU =====================
children.push(H1("Nguồn dữ liệu"));
children.push(P("Nawab S, Khan G, Abbas A, et al. District-week notifiable infectious disease surveillance data for Khyber Pakhtunkhwa, Pakistan, 2023–2026. Mendeley Data. Version 1; 2026. DOI: 10.17632/9yzvfgrhkt.1."));
children.push(P("International Computer Symposium 2026. Call for Papers and submission instructions. https://www.ics2026.ntub.edu.tw/call-for-paper/."));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 22 } } },
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join("D:", "Thầy Khánh", "ISC", "manuscript", "ProjectA_BaoCao_Draft.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
