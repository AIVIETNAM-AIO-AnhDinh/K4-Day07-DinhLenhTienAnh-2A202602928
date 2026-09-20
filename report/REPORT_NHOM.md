# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** 5changlinhngulam
**Thành viên:** Nguyễn Hoàng Nam; Đinh Lệnh Tiến Anh; Vũ Hải Minh
**Ngày:** 20/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Chính sách thương mại điện tử Shopee dành cho Người Mua và Người Bán.

**Tại sao nhóm chọn chủ đề này?**
Nhóm chọn chủ đề này vì corpus gồm các chính sách công khai có cấu trúc và điều kiện cụ thể,
phù hợp để kiểm tra chunking, metadata filtering và khả năng truy xuất của hệ thống RAG.
Việc tách `audience=buyer` và `audience=seller` cũng tạo ra tình huống thực tế để đánh giá
liệu metadata filter có giúp tránh trả lời nhầm đối tượng hay không.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | `shopee-dispute-resolution.md` | https://help.shopee.vn/portal/4/article/77265 | 2026-09-20 / not-stated | 5,012 | `doc_id=shopee-dispute-resolution`; `audience=buyer`; `category=dispute`; `language=vi` |
| 2 | `shopee-listing-policy.md` | https://help.shopee.vn/portal/4/article/77246 | 2026-09-20 / not-stated | 22,076 | `doc_id=shopee-listing-policy`; `audience=seller`; `category=listing-policy`; `language=vi` |
| 3 | `shopee-prohibited-items.md` | https://help.shopee.vn/portal/4/article/77247 | 2026-09-20 / not-stated | 13,277 | `doc_id=shopee-prohibited-items`; `audience=seller`; `category=listing-policy`; `language=vi` |
| 4 | `shopee-return-policy.md` | https://help.shopee.vn/portal/4/article/77251 | 2026-09-20 / not-stated | 19,936 | `doc_id=shopee-return-policy`; `audience=buyer`; `category=returns-policy`; `language=vi` |
| 5 | `shopee-shipping-policy.md` | https://help.shopee.vn/portal/4/article/77250 | 2026-09-20 / not-stated | 25,113 | `doc_id=shopee-shipping-policy`; `audience=seller`; `category=shipping`; `language=vi` |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `shopee-return-policy` | Định danh ổn định, dùng để truy vết chunk về tài liệu gốc và hỗ trợ xóa tài liệu. |
| `title` | string | `Chính sách Trả hàng và Hoàn tiền` | Giúp nhận diện chủ đề và hiển thị nguồn dễ đọc. |
| `source_url` | URL | `https://help.shopee.vn/portal/4/article/77251` | Truy vết câu trả lời về nguồn công khai ban đầu. |
| `retrieved_at` | date | `2026-09-20` | Kiểm tra độ mới của dữ liệu và thời điểm thu thập. |
| `document_version` | string | `not-stated` | Phân biệt phiên bản hoặc ngày hiệu lực của chính sách. |
| `audience` | enum | `buyer` / `seller` | Lọc kết quả theo đối tượng; corpus hiện có 2 buyer và 3 seller. |
| `category` | enum | `returns-policy` / `shipping` | Lọc theo loại chính sách để giảm kết quả không liên quan. |
| `language` | string | `vi` | Lọc tài liệu theo ngôn ngữ truy vấn hoặc người dùng. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `shopee-dispute-resolution.md` | FixedSizeChunker (`fixed_size`) | 24 | 195.9 | Một phần; có thể cắt giữa câu/heading |
| `shopee-dispute-resolution.md` | SentenceChunker (`by_sentences`) | 10 | 468.0 | Tốt ở ranh giới câu, nhưng độ dài không đều |
| `shopee-dispute-resolution.md` | RecursiveChunker (`recursive`) | 31 | 150.3 | Khá tốt; ưu tiên separator lớn rồi mới cắt nhỏ |
| `shopee-listing-policy.md` | FixedSizeChunker (`fixed_size`) | 107 | 199.9 | Một phần; kích thước đều nhưng có thể cắt ngữ nghĩa |
| `shopee-listing-policy.md` | SentenceChunker (`by_sentences`) | 78 | 271.4 | Tốt ở ranh giới câu |
| `shopee-listing-policy.md` | RecursiveChunker (`recursive`) | 143 | 147.6 | Khá tốt; giữ separator nhưng tạo nhiều chunk hơn |
| `shopee-prohibited-items.md` | FixedSizeChunker (`fixed_size`) | 64 | 198.8 | Một phần; có thể cắt giữa câu/heading |
| `shopee-prohibited-items.md` | SentenceChunker (`by_sentences`) | 55 | 228.8 | Tốt ở ranh giới câu, độ dài không hoàn toàn đều |
| `shopee-prohibited-items.md` | RecursiveChunker (`recursive`) | 83 | 151.3 | Khá tốt; cân bằng ngữ cảnh và kích thước |

### Chiến lược của từng thành viên

> Mỗi thành viên điền một khối dưới đây (copy thêm nếu nhóm có nhiều hơn 3 người).

**Thành viên 1 — Nguyễn Hoàng Nam**
- **Loại chiến lược:** `FixedSizeChunker` có overlap
- **Mô tả & lý do chọn:** Chia văn bản theo cửa sổ ký tự cố định và giữ overlap để các ý nằm ở ranh giới chunk vẫn có cơ hội xuất hiện ở hai chunk liên tiếp. Chiến lược đơn giản, dễ kiểm soát kích thước nhưng có thể cắt giữa câu hoặc heading.

**Thành viên 2 — Đinh Lệnh Tiến Anh**
- **Loại chiến lược:** `RecursiveChunker`
- **Mô tả & lý do chọn:** Ưu tiên các ranh giới lớn như đoạn văn, dòng mới và câu, sau đó mới hạ xuống separator nhỏ hơn khi cần. Cách này giữ ngữ nghĩa tốt hơn fixed-size và vẫn kiểm soát được kích thước chunk.

**Thành viên 3 — Vũ Hải Minh**
- **Loại chiến lược:** Custom `MarkdownSectionChunker` theo heading/section
- **Mô tả & lý do chọn:** Văn bản chính sách đã được biên soạn theo các heading, nên mỗi section là một đơn vị ngữ nghĩa tự nhiên. Section dài được chia tiếp bằng `RecursiveChunker`, đồng thời gắn lại heading vào từng chunk con để không mất ngữ cảnh. Implementation nằm trong `bench.py`.
- **Code snippet (nếu custom):**
```python
for section in sections:
    heading, body = split_heading(section)
    pieces = RecursiveChunker(chunk_size=available_size).chunk(body)
    chunks.extend(f"{heading}\n{piece}" for piece in pieces)
```

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Hoàng Nam | FixedSizeChunker có overlap | 1/5 theo report, MockEmbedder | Dễ triển khai, kích thước và overlap rõ ràng | Có thể cắt gãy câu/heading; kết quả mock thấp |
| Đinh Lệnh Tiến Anh | RecursiveChunker | 5/5 theo report, LocalEmbedder | Giữ ranh giới ngữ nghĩa và đạt kết quả tốt với embedding thật | Tốn chi phí/tài nguyên hơn; kết quả không so sánh trực tiếp với MockEmbedder |
| Vũ Hải Minh | MarkdownSectionChunker theo heading | 0/5 ở mức content, 1/5 ở mức doc_id với MockEmbedder | Giữ tiêu đề và cấu trúc section; các chunk con vẫn có heading | MockEmbedder chọn sai section dù có thể chọn đúng tài liệu |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
Trong các kết quả đã báo cáo, `RecursiveChunker` kết hợp với `LocalEmbedder` cho kết quả tốt nhất
với 5/5 câu có chunk liên quan trong top-3. Tuy nhiên không nên kết luận chỉ từ điểm số này vì
FixedSize và heading trong các report còn dùng MockEmbedder; để so sánh công bằng, nhóm cần chạy
cùng một backend thật và cùng bộ query. Về mặt cấu trúc chính sách, chiến lược heading vẫn hữu ích
vì bảo toàn tiêu đề, và có thể kết hợp heading với recursive splitting thành chiến lược hybrid.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Người Mua có bao nhiêu ngày để gửi yêu cầu trả hàng/hoàn tiền kể từ khi đơn hàng được cập nhật giao hàng thành công? | 15 ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công; riêng thực phẩm tươi sống và đông lạnh là trong vòng 24 giờ. | `shopee-return-policy`, Mục 3.2 |
| 2 | Người Mua được quyền yêu cầu trả hàng/hoàn tiền trong những trường hợp nào? | Không nhận được/nhận không đủ sản phẩm; hàng giả, hàng nhái; sản phẩm lỗi hoặc hư hại khi vận chuyển; giao sai sản phẩm; khác biệt rõ rệt với mô tả; hết hạn; Người Bán đồng ý; hoặc trả hàng COM. | `shopee-return-policy`, Mục 3.1 |
| 3 | Quy trình giải quyết tranh chấp/khiếu nại của Shopee gồm những bước nào? | Gồm 4 bước: Người Mua gửi khiếu nại; bộ phận khiếu nại tiếp nhận; Shopee xử lý theo chính sách hoặc đưa hướng giải quyết trong 07 ngày làm việc; vụ việc ngoài thẩm quyền được đưa đến cơ quan nhà nước có thẩm quyền. | `shopee-dispute-resolution`, Mục 1, Bước 1–4 |
| 4 | Người Bán vi phạm Chính Sách Cấm/Hạn Chế Sản Phẩm có thể bị áp dụng những chế tài nào? | Xóa sản phẩm; giới hạn quyền tài khoản; đình chỉ hoặc xóa tài khoản; cấn trừ số dư/phong tỏa quyền rút tiền; và các chế tài pháp lý hoặc bồi thường khác. | `shopee-prohibited-items`, Mục 3 |
| 5 | Shopee xử lý khiếu nại trong bao nhiêu ngày làm việc? | Với Người Mua, tranh chấp không phải khiếu nại Trả Hàng/Hoàn Tiền được đưa hướng giải quyết trong vòng 07 ngày làm việc kể từ khi nhận đủ thông tin/tài liệu. Câu này cần `metadata_filter={"audience": "buyer"}`. | `shopee-dispute-resolution`, Mục 1, Bước 3 |

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Thời hạn gửi yêu cầu trả hàng/hoàn tiền | RecursiveChunker + LocalEmbedder | Có | Top-1 chứa mốc 15 ngày theo report của Tiến Anh; Fixed/heading dùng mock không lấy được đúng section. |
| 2 | Điều kiện được trả hàng/hoàn tiền | RecursiveChunker + LocalEmbedder | Có | Chunk recursive chứa danh sách điều kiện trả hàng. |
| 3 | Quy trình giải quyết tranh chấp/khiếu nại | RecursiveChunker + LocalEmbedder | Có | Truy xuất được tài liệu dispute và quy trình 4 bước. |
| 4 | Chế tài khi vi phạm chính sách cấm | RecursiveChunker + LocalEmbedder | Có | Chunk chứa phần biện pháp chế tài. |
| 5 | Thời hạn xử lý khiếu nại, có filter buyer | RecursiveChunker + LocalEmbedder | Có | Filter giữ đúng audience buyer; gold chunk ở top-2 theo report Tiến Anh. |

> **Lưu ý về tính công bằng của phép so sánh:** bảng trên tổng hợp các lần chạy được ghi trong
> hai report cá nhân, không phải một thí nghiệm A/B dùng cùng embedding backend. Report của Đinh
> Lệnh Tiến Anh có một lần chạy `RecursiveChunker + LocalEmbedder` đạt 5/5; các kết quả của
> Nguyễn Hoàng Nam và Vũ Hải Minh được ghi với `MockEmbedder`. Vì vậy, kết luận về chất lượng
> retrieval chỉ nên xem là kết quả quan sát ban đầu; cần chạy lại cả ba chiến lược với cùng
> backend thật để xếp hạng định lượng công bằng.

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
Có, rõ nhất ở câu 5. Khi không lọc, top-3 có thể trộn tài liệu `buyer` và `seller`, trong đó
chính sách vận chuyển của seller có mốc 10 ngày làm việc khác với mốc 07 ngày của buyer.
Khi dùng `metadata_filter={"audience": "buyer"}`, toàn bộ ứng viên thuộc đúng đối tượng và
gold chunk được cải thiện thứ hạng. Tuy nhiên filter chỉ kiểm soát đối tượng, không bảo đảm chunk
đang chứa đúng section hoặc số liệu cần trả lời.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
- Kiểm tra `doc_id` chưa đủ: một chunk có thể thuộc đúng tài liệu nhưng không chứa gold answer; cần chấm thêm ở mức nội dung.
- `MockEmbedder` tạo vector từ MD5 nên kết quả similarity là nhiễu; điểm 5/5 của RecursiveChunker chỉ có ý nghĩa khi ghi rõ đã dùng `LocalEmbedder`.
- Metadata filter là lớp bảo vệ cứng cho các câu hỏi mơ hồ về buyer/seller, nhưng không thay thế chunking và embedding ngữ nghĩa.

**Bài học rút ra khi so sánh trong nhóm:**
Fixed-size dễ kiểm soát nhưng có thể cắt gãy ngữ cảnh. Recursive giữ các ranh giới tự nhiên tốt hơn,
còn heading-based bảo toàn cấu trúc section và tiêu đề. Kết quả retrieval phụ thuộc đồng thời vào
chunking, embedding backend và metadata, nên mọi thành viên phải dùng cùng thiết lập khi so sánh.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
Nhóm sẽ dùng embedding đa ngôn ngữ thật cho toàn bộ chiến lược, cache embedding theo hash và
giữ một bộ đánh giá content-level với các chuỗi gold bắt buộc. Với tài liệu chính sách, nhóm sẽ
dùng hybrid heading-plus-recursive để vừa giữ tiêu đề vừa giới hạn kích thước các section dài.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 13 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 7 / 10 |
| Thuyết trình (Demo) | 4 / 5 |
| **Tổng phần nhóm** | **34 / 40** |
