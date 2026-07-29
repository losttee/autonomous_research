# Roadmap phát triển — Autonomous Research & Decision Agent

Lộ trình phát triển và hoàn thiện hệ thống Autonomous Research Agent: tập trung vào tính tự chủ, kiểm chứng nguồn (grounding), tối ưu chi phí và trải nghiệm người dùng thực tế.

---

## Milestone 1: Hoàn thiện Core Pipeline & Trả nợ kỹ thuật

- [x] **Vòng lặp `MAX_REPLAN` linh hoạt**: `core/pipeline.py` chạy vòng lặp re-plan khi dữ liệu thu thập quá mỏng (thin evidence), hỗ trợ tham số `max_replan` và chuỗi `previous_plan_id` giữa các revision.
- [x] **Cấu hình giá model gateway**: Biến môi trường `MODEL_PRICING` (`model:in:out,...`) cho phép override bảng giá token trong `core/config.py` mà không cần sửa code; tự động bỏ qua entry lỗi an toàn.
- [x] **Đồng bộ Dependencies**: Dọn dẹp các thư viện thử nghiệm không dùng khỏi `pyproject.toml`, chuẩn hóa runtime dependencies khớp hoàn toàn với `requirements.txt`.

---

## Milestone 2: Evaluation Harness — Đo lường chất lượng & Grounding

Không có lớp đo lường thì mọi cải tiến về sau đều là phỏng đoán. Đây là phần bảo chứng giá trị của toàn bộ kiến trúc verifier.

- [x] **Golden set**: `evaluation/golden_set.json` — 12 câu phủ 3 dạng: factual (6 câu kèm expect_keywords), open (3 câu), unanswerable (3 câu yêu cầu agent nhận thức sự không chắc chắn).
- [x] **Metrics tự động** trong `evaluation/metrics.py`:
      - *Citation integrity*: mọi `[source_id]` trong report phải resolve đúng nguồn thật (100%, kiểm tra deterministic không cần LLM).
      - *Grounding precision*: sample claims (seed cố định) → LLM-judge độc lập chấm lại grounding, fail-closed khi judge lỗi; tắt được bằng `--no-judge`.
      - *Honesty*: câu unanswerable phải rơi vào `uncertainties` hoặc confidence thấp.
      - *Keyword recall*: expect_keywords xuất hiện trong report (có nghĩa với search backend thật).
      - *Cost/latency* mỗi câu ghi nhận từ `CostSnapshot`.
- [x] **Runner**: `evaluation/run_eval.py` → xuất file `evaluation/results/<timestamp>.json` + bảng summary; `evaluation/compare.py` so sánh diff giữa hai lần chạy.

---

## Milestone 3: Observability & Trải nghiệm thời gian thực (SSE)

- [x] **Log Aggregator**: `monitoring/aggregate.py` đọc JSON-lines log (`LOG_FILE`, mặc định `data/logs/pipeline.jsonl`) → rollup theo tầng pipeline + danh sách lần chạy gần đây.
- [x] **Endpoints & Dashboard**:
      - `GET /metrics`: API trả về JSON tổng hợp chi phí/độ trễ.
      - `GET /monitoring`: Giao diện dashboard HTML theo dõi trực quan chi phí theo tầng, lịch sử chạy và cảnh báo chạm trần ngân sách.
- [x] **Streaming tiến trình (SSE)**: `POST /research/stream` phát sự kiện từng tầng ("Đang lập kế hoạch → Đang thu thập → Đang đối chiếu → Đang tổng hợp") rồi trả kết quả cuối, dùng `StreamingResponse` thuần của Starlette (không phát sinh dependency dư thừa). Frontend tự động fallback về `/research` nếu stream bị gián đoạn.
- [x] **Cải tiến UI**: Hiển thị `contradictions` và gắn badge `[bộ nhớ]` cho các nguồn recall từ vector store.
- [x] **Test suite**: Bổ sung bộ kiểm thử riêng cho log aggregation, thứ tự progress events, và SSE stream.

---

## Milestone 4: Mở rộng Tooling (MCP Servers)

Hiện thực hóa `executor/mcp_servers/`. Giữ đúng pattern `SearchTool`: interface cố định, implementation có thể thay thế, có stub cho unit test.

- [ ] **Calculator tool**: Xử lý số liệu tài chính/toán học chính xác tuyệt đối, tránh để LLM nhẩm; kết quả trích dẫn dưới dạng `SourceRef(type=CALCULATOR)`.
- [ ] **Document reader tool**: Đọc tài liệu cục bộ (PDF/Markdown) làm `SourceType.INTERNAL_RAG`, cho phép nghiên cứu trên tài liệu riêng.
- [ ] Planner nhận danh sách tool khả dụng trong prompt để sinh `tool_hint` chính xác; executor dispatch theo hint tương ứng.
- [ ] Giới hạn budget riêng cho từng tool tính vào `tracker.record_tool_call()`.

---

## Milestone 5: Nâng cao năng lực Verifier

- [ ] **Phát hiện mâu thuẫn chéo giữa các sub-task**: Mở rộng verifier không chỉ kiểm tra claim-vs-nguồn mà còn đối chiếu "2 sub-task khác nhau cùng nói về 1 thực thể nhưng số liệu lệch nhau".
- [ ] **Dedup/Merge claims**: Gom cụm các nhận định tương đồng trước khi chuyển vào synthesizer — giảm tiêu thụ token và tránh lặp ý trong báo cáo.
- [ ] **Adversarial Verification**: Với các nhận định quan trọng, chạy thêm một lượt phản biện yêu cầu model tìm bằng chứng bác bỏ; chỉ giữ lại khi đạt đồng thuận cao.

---

## Milestone 6: Hoàn thiện cơ chế Memory & Caching

- [ ] **Deduplication trong Memory**: Tránh lưu trữ các claim trùng lặp khi chạy lại cùng chủ đề (kiểm tra cosine similarity ≥ 0.98 trước khi add).
- [ ] **Freshness & TTL**: Áp dụng trọng số thời gian (`SourceRef.retrieved_at`) — giảm độ tin cậy của thông tin giá cả/số liệu cũ khi recall.
- [ ] Đánh giá độ hữu ích của recall bằng chính evaluation harness để tinh chỉnh ngưỡng cosine threshold.
- [ ] Cân nhắc tích hợp Redis / ChromaDB khi file lưu trữ vector JSON cục bộ vượt ngưỡng kích thước lớn.

---

## Milestone 7: Hardening & Benchmark công khai

- [ ] **Error-injection tests**: Kiểm thử khả năng chịu lỗi khi mất kết nối mạng, LLM trả về format sai, hoặc cạn budget giữa chừng — đảm bảo mọi degradation path đều trả về báo cáo hợp lệ.
- [ ] **Benchmark so sánh**: Đánh giá trên golden set giữa bản đầy đủ và bản tắt verifier để định lượng giá trị của cơ chế grounding.
- [ ] Đóng gói phiên bản ổn định và cập nhật README với kết quả benchmark chính thức.

---

## Bảng mức độ ưu tiên

| Ưu tiên | Hạng mục | Mục tiêu |
|---|---|---|
| 1 | Core & Replan | Ổn định luồng xử lý và tối ưu quản lý ngân sách |
| 2 | Evaluation Harness | Bộ công cụ đo lường khách quan, làm cơ sở cho mọi cải tiến |
| 3 | Observability & SSE | Giám sát chi phí thời gian thực và nâng cao trải nghiệm |
| 4 | Verifier sâu | Điểm khác biệt cốt lõi: triệt tiêu ảo giác và đối chiếu mâu thuẫn |
| 5 | Tooling mở rộng | Bổ sung công cụ tính toán và đọc tài liệu cục bộ |
| 6 | Memory & TTL | Tối ưu tái sử dụng tri thức và loại bỏ dữ liệu lỗi thời |
