# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đinh Lệnh Tiến Anh

**Nhóm:** 5changlinhngulam

**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:* có nghĩa là 2 vector chỉ hướng gần giống nhau. trong nlp, điều này có nghĩa là 2 đối tượng văn bản có nghĩa gần giống nhau

**Ví dụ có độ tương tự CAO:**
- Câu A: tôi thích học AI
- Câu B: tôi thích học ML
- Tại sao tương đồng: đều biểu hiện yêu thích đối với 2 chủ đề có liên quan mật thiết đến nhau

**Ví dụ có độ tương tự THẤP:**
- Câu A: mô hình AI được tối ưu bằng hàm loss
- Câu B: hôm nay tôi đi bơi
- Tại sao khác: 2 câu không chia sẻ chung ngữ cảnh

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:* khi dùng euclidean distance, văn bản dài thường xa hơn do mang nhiều từ. điều này không phản ánh ngữ nghĩa

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* step = chunk_size - overlap = 500 - 50 = 450
> số_chunk = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11) = 23
> *Đáp án:* 23

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:* step còn 400 nên số chunk tăng: `ceil((10,000 - 100) / 400) = ceil(24.75) = 25 chunks` (tốn thêm chi phí nhúng và dung lượng vector store).
> Ta vẫn muốn overlap lớn hơn vì nó giữ ngữ cảnh ở ranh giới chunk. một câu hoặc một ý bị cắt đôi vẫn xuất hiện trọn vẹn trong ít nhất một chunk, nhờ đó truy xuất không bỏ sót thông tin nằm vắt qua hai chunk.

---


## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])\s+` — một **lookbehind**, nên dấu câu được giữ lại ở cuối câu thay vì bị nuốt mất như khi `split(". ")`; `\s+` nuốt luôn cả khoảng trắng lẫn xuống dòng nên xử lý được cả `". "` và `".\n"` bằng một biểu thức duy nhất.
> Sau khi tách, tôi `strip()` từng câu và loại bỏ câu rỗng, rồi gom lại theo từng nhóm `max_sentences_per_chunk` câu bằng slicing và nối lại bằng một dấu cách.
> Edge case đã xử lý: chuỗi rỗng hoặc chỉ toàn khoảng trắng → trả về `[]`; khoảng trắng thừa ở cuối văn bản (tạo ra phần tử rỗng sau khi split) → bị lọc bỏ; `max_sentences_per_chunk` ≤ 0 → đã được `__init__` ép về tối thiểu là 1 nên không bao giờ chia cho 0 hay lặp vô hạn.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split` thử các dấu phân cách theo thứ tự ưu tiên từ "mạnh" đến "yếu" (`\n\n` → `\n` → `". "` → `" "` → `""`): cắt văn bản bằng dấu phân cách hiện tại, rồi **gom lại (greedy merge)** các mảnh liền kề vào một bộ đệm chừng nào tổng độ dài còn ≤ `chunk_size`. Ý tưởng là ưu tiên giữ nguyên ranh giới ngữ nghĩa lớn nhất có thể, chỉ cắt nhỏ hơn khi thực sự cần.
> Mảnh nào tự nó vẫn dài hơn `chunk_size` thì được **đệ quy** với phần đuôi danh sách separator (dấu phân cách yếu hơn) — đây là bước làm nhỏ dần bài toán.
> Có 3 base case: (1) text rỗng → `[]`; (2) `len(text) <= chunk_size` → trả về nguyên khối, không cắt nữa; (3) hết separator (hoặc gặp `""`) → `_hard_split` cắt cứng theo độ dài. Base case (3) bảo đảm đệ quy luôn dừng kể cả khi văn bản không chứa bất kỳ dấu phân cách nào. Nếu separator hiện tại không xuất hiện trong text (`split` trả về đúng 1 mảnh), tôi bỏ qua và đệ quy thẳng sang separator kế tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi tách riêng một helper `_make_record` để chuẩn hoá mỗi document thành một record thống nhất `{id, doc_id, content, embedding, metadata}`, nhờ đó cả hai backend (in-memory và ChromaDB) dùng chung một định dạng. Mỗi record được gán id duy nhất dạng `"{doc.id}#{self._next_index}"` — điều này quan trọng vì nếu nạp cùng một `doc.id` hai lần thì store phải đếm thành 2 chunk chứ không ghi đè (ChromaDB cũng bắt buộc id duy nhất). Tôi luôn nhét thêm `doc_id` vào metadata để `delete_document` có thể truy ngược.
> `search` nhúng câu truy vấn rồi tính **tích vô hướng (dot product)** với từng embedding đã lưu, sắp xếp giảm dần và cắt lấy `top_k`. Vì `MockEmbedder` (cũng như OpenAI/Gemini/SentenceTransformers với `normalize_embeddings=True`) trả về vector đã chuẩn hoá về độ dài 1, nên dot product ở đây **bằng đúng cosine similarity** mà không tốn phép chia — đây là lý do vector store thực tế hay chuẩn hoá trước rồi chỉ dùng dot product.
> Ở nhánh ChromaDB tôi tạo collection với `metadata={"hnsw:space": "cosine"}` và quy đổi `score = 1 - distance`, để điểm số của hai backend nằm trên cùng một thang đo.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi **lọc trước, tìm kiếm sau (pre-filtering)**: lọc danh sách record theo `metadata_filter` (khớp tất cả cặp key-value), rồi mới chạy tìm kiếm tương tự trên tập ứng viên còn lại qua `_search_records`. Lọc trước tốt hơn lọc sau vì nếu tìm top-k trước rồi mới lọc thì rất dễ trả về ít hơn k kết quả (thậm chí rỗng) khi các chunk điểm cao đều bị loại — pre-filtering bảo đảm luôn lấy đủ k chunk tốt nhất *trong phạm vi hợp lệ*. Khi `metadata_filter` là `None`, hàm chạy y hệt `search` bình thường.
> `delete_document` xoá theo `metadata['doc_id']` chứ không theo id của chunk, nên **một lần gọi xoá sạch mọi chunk thuộc cùng tài liệu**. Tôi build lại list các record còn lại và so sánh độ dài trước/sau để quyết định trả về `True` (có xoá được) hay `False` (không tìm thấy `doc_id`). Nhánh ChromaDB dùng `collection.get(where={"doc_id": ...})` rồi `collection.delete(ids=...)`.

> **Ghi chú về backend:** `EmbeddingStore` thử `import chromadb` trước, nếu không có thì tự động rơi về store in-memory. Trong môi trường của tôi chưa cài `chromadb` nên toàn bộ kết quả dưới đây chạy trên nhánh in-memory; nhánh ChromaDB vẫn được lập trình đầy đủ (init client + collection, add, query, filter `$and`, delete) để chạy được khi cài thêm thư viện.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `answer` thực hiện đúng 3 bước của mẫu RAG: (1) `store.search(question, top_k)` để truy xuất chunk liên quan, (2) dựng prompt, (3) gọi `llm_fn(prompt)`. Prompt được chia thành các khối có nhãn rõ ràng — `SYSTEM_INSTRUCTION` → `=== CONTEXT ===` → `=== QUESTION ===` → `=== ANSWER ===` — để mô hình phân biệt được đâu là dữ liệu tham chiếu, đâu là câu hỏi.
> Ngữ cảnh được đưa vào dưới dạng danh sách đánh số `[1] (source: ...)` kèm nguồn lấy từ metadata, nhằm hai mục đích: buộc mô hình trích dẫn được chunk nào đã dùng, và giúp tôi truy vết câu trả lời sai về đúng tài liệu gốc.
> Để giảm ảo giác (hallucination), chỉ thị hệ thống yêu cầu **chỉ trả lời dựa trên ngữ cảnh** và nói "không biết" nếu ngữ cảnh không chứa câu trả lời; trường hợp store không trả về chunk nào, tôi chèn chuỗi `(No relevant context was found...)` thay vì để phần context trống — nếu để trống, mô hình sẽ có xu hướng bịa từ kiến thức nền.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0 -- /Users/tienanh211/K4-Day07-DinhLenhTienAnh-2A202602928/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/tienanh211/K4-Day07-DinhLenhTienAnh-2A202602928
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.02s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

> **Backend đã dùng:** `MockEmbedder` (mặc định). 

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | tôi thích học AI | tôi thích học ML | cao | **-0.0292** | ❌ |
| 2 | Con mèo đang ngủ trên ghế sofa. | Chú mèo nằm thiu thiu trên chiếc đi văng. | cao | **+0.1366** | ✅ |
| 3 | Chính sách đổi trả cho phép hoàn hàng trong 7 ngày. | Thời hạn trả lại sản phẩm là một tuần. | cao | **-0.0626** | ❌ |
| 4 | mô hình AI được tối ưu bằng hàm loss | hôm nay tôi đi bơi | thấp | **-0.1115** | ✅ |
| 5 | Vector database stores embeddings for similarity search. | Đội bóng đã ghi ba bàn trong hiệp hai. | thấp | **-0.1586** | ✅ |

**Số dự đoán đúng: 3/5 — đúng bằng mức đoán ngẫu nhiên.**

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là **cặp 1**: hai câu chỉ khác nhau đúng một từ ("AI" và "ML") mà lại nhận điểm **âm** (-0.0292). Nếu chỉ nhìn bảng, ta sẽ kết luận sai rằng "AI" và "ML" không liên quan gì nhau. Tương tự, cặp 3 diễn đạt cùng một chính sách ("7 ngày" = "một tuần") nhưng điểm (-0.0626) còn *thấp hơn* cặp 4 vốn là hai câu chẳng ăn nhập gì (-0.1115 thì thấp hơn, nhưng khoảng cách giữa chúng quá nhỏ để có ý nghĩa). Nói cách khác, thứ hạng gần như vô nghĩa.
> Nguyên nhân không phải do cài đặt sai, mà do `MockEmbedder` sinh vector bằng **md5 của chuỗi ký tự** rồi bung ra qua bộ sinh số giả ngẫu nhiên. Tôi kiểm chứng bằng hai phép đo đối chứng:
> - Hai chuỗi **giống hệt nhau** → similarity = **1.0000** (vì cùng hash).
> - `"xin chào"` với `"xin chào."` — **chỉ khác đúng một dấu chấm** → similarity = **0.0236**, tức gần như vuông góc.
>
> Hash embedding chỉ làm được phép **so khớp chính xác**, thêm một ký tự là toàn bộ vector đổi hoàn toàn. Một mô hình embedding thật phải có tính chất ngược lại — thay đổi nhỏ về mặt chữ viết chỉ gây thay đổi nhỏ về vector, còn thay đổi lớn về *ý nghĩa* mới gây thay đổi lớn. Chính tính **liên tục giữa không gian ngữ nghĩa và không gian vector** đó mới làm retrieval hoạt động được; bản thân công thức cosine **không tạo ra ngữ nghĩa**, nó chỉ đo lại thứ ngữ nghĩa mà mô hình nhúng đã đặt sẵn vào vector.
> `compute_similarity` đúng (kiểm chứng bằng vector trùng nhau → 1.0, trực giao → 0.0, đối hướng → -1.0; 4/4 test pass), nhưng muốn có số liệu benchmark có ý nghĩa thì **phải đổi sang backend nhúng thật**. Dùng `paraphrase-multilingual-MiniLM-L12-v2` phù hợp với dữ liệu tiếng Việt hơn.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

> **Thiết lập:** 5 tài liệu chính sách công khai của Shopee trong `data/ecommerce/` → `MarkdownSectionChunker(chunk_size=700)` → **156 chunk** → `EmbeddingStore` (in-memory, `MockEmbedder`) → `KnowledgeBaseAgent`.
> **Script tái lập:** `python bench.py` — in ra số chunk đã nạp và top-3 cho cả 5 câu.

**Phân bố chunk theo tài liệu:**

| doc_id | Số chunk | audience | Nguồn |
|--------|----------|----------|-------|
| `shopee-dispute-resolution` | 10 | buyer | help.shopee.vn/portal/4/article/77265 |
| `shopee-listing-policy` | 37 | seller | help.shopee.vn/portal/4/article/77246 |
| `shopee-prohibited-items` | 22 | seller | help.shopee.vn/portal/4/article/77247 |
| `shopee-return-policy` | 42 | buyer | help.shopee.vn/portal/4/article/77251 |
| `shopee-shipping-policy` | 45 | seller | help.shopee.vn/portal/4/article/77250 |
| **Tổng** | **156** | | *retrieved_at: 2026-09-20* |

**Bộ 5 câu hỏi đánh giá** — đa dạng dạng hỏi, mỗi câu có gold answer **trích được trực tiếp** từ tài liệu (không suy đoán chính sách của nền tảng):

| # | Dạng hỏi | Câu hỏi | Gold answer (trích từ tài liệu) | Nguồn |
|---|----------|---------|--------------------------------|-------|
| 1 | Tra số liệu | Người Mua có bao nhiêu ngày để gửi yêu cầu trả hàng/hoàn tiền kể từ khi đơn hàng được cập nhật giao hàng thành công? | **15 (mười lăm) ngày**; riêng thực phẩm tươi sống và đông lạnh: **trong vòng 24 giờ** | `shopee-return-policy` §3.2 |
| 2 | Hỏi điều kiện | Người Mua được quyền yêu cầu trả hàng/hoàn tiền trong những trường hợp nào? | Không nhận được SP / không nhận đủ / nhận hàng giả-nhái; SP lỗi hoặc hư hại khi vận chuyển; giao sai SP; SP khác biệt rõ rệt so với mô tả; SP hết hạn sử dụng; Người Bán đã thỏa thuận đồng ý; Trả hàng COM | `shopee-return-policy` §3.1 |
| 3 | Hỏi quy trình | Quy trình giải quyết tranh chấp/khiếu nại của Shopee gồm những bước nào? | **4 bước**: B1 bấm khiếu nại trong mục "Đơn Mua" → B2 bộ phận khiếu nại tiếp nhận → B3 xử lý theo Chính Sách Trả Hàng Hoàn Tiền (tranh chấp khác: 07 ngày làm việc) → B4 ngoài thẩm quyền thì chuyển cơ quan nhà nước | `shopee-dispute-resolution` §1 |
| 4 | Liệt kê | Người Bán vi phạm Chính Sách Cấm/Hạn Chế Sản Phẩm có thể bị áp dụng những chế tài nào? | (i) Sản phẩm bị xóa; (ii) Tài khoản bị giới hạn quyền; (iii) Tài khoản bị đình chỉ/xóa; (iv) Cấn trừ số dư, phong tỏa quyền rút tiền; (v) Chế tài khác: phạt hành chính, xử lý hình sự, bồi thường thiệt hại | `shopee-prohibited-items` §3 |
| 5 | Tra số liệu — **cần `metadata_filter`** | Shopee xử lý khiếu nại trong bao nhiêu ngày làm việc? | Với **Người Mua**: **07 ngày làm việc** kể từ ngày nhận đủ thông tin/tài liệu (tranh chấp không phải khiếu nại Trả Hàng/Hoàn Tiền) | `shopee-dispute-resolution` §1 Bước 3 |

**Kết quả truy xuất thực tế** (`python bench.py`):

| # | Câu hỏi (rút gọn) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn gửi yêu cầu trả hàng? | `shopee-dispute-resolution` — đoạn về Người tiêu dùng dễ bị tổn thương | +0.2609 | ❌ Không | Tổng hợp 3 chunk, không chứa mốc 15 ngày |
| 2 | Điều kiện được trả hàng/hoàn tiền? | `shopee-listing-policy` — quảng cáo gây nhầm lẫn | +0.3429 | ❌ Không | Ngữ cảnh lệch sang quy định đăng bán |
| 3 | Quy trình giải quyết tranh chấp? | `shopee-return-policy` — giới hạn hạn mức | +0.2898 | ❌ Không | Không nêu được 4 bước |
| 4 | Chế tài khi vi phạm chính sách cấm? | `shopee-return-policy` — hoàn phí vận chuyển "Tự sắp xếp" | +0.4706 | ❌ Không | Ngữ cảnh sai hoàn toàn |
| 5 | Xử lý khiếu nại bao nhiêu ngày? (**có lọc** `audience=buyer`) | `shopee-dispute-resolution` — "Bước 1: Để tạo khiếu nại..." (hạng #3) | +0.2055 | ✅ Có | Ngữ cảnh đúng đối tượng Người Mua |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **1 / 5**

### Câu 5 — vì sao bắt buộc phải có `metadata_filter`

Câu hỏi *"Shopee xử lý khiếu nại trong bao nhiêu ngày làm việc?"* **cố tình không nêu rõ người hỏi là ai**, trong khi corpus chứa đúng hai tài liệu cùng chủ đề, **dùng chung từ vựng** ("khiếu nại", "ngày làm việc"), nhưng **khác đối tượng và khác đáp án**:

| Tài liệu | audience | Đáp án |
|----------|----------|--------|
| `shopee-dispute-resolution` | **buyer** | **07 ngày làm việc** (tranh chấp không phải trả hàng/hoàn tiền) |
| `shopee-shipping-policy` | **seller** | **tối đa 10 ngày làm việc** (khiếu nại về vận chuyển) |

Kết quả chạy thật cho thấy đúng hai hành vi khác nhau:

- **Không lọc** → top-3 trả về audience `['buyer', 'seller']` — **lẫn hai đối tượng**. Agent hoàn toàn có thể lấy mốc 10 ngày (dành cho Người Bán) để trả lời một Người Mua, tức **trả lời sai đối tượng** dù trích dẫn vẫn "có trong tài liệu".
- **Có lọc `metadata_filter={"audience": "buyer"}`** → top-3 toàn bộ là `['buyer']`, và chunk gold của `shopee-dispute-resolution` leo lên hạng #3.

> **Điểm quan trọng nhất rút ra:** hiệu quả của `metadata_filter` **không phụ thuộc vào chất lượng embedding**. Đây là ràng buộc cứng trên metadata, nên nó vẫn bảo đảm đúng đối tượng ngay cả khi xếp hạng ngữ nghĩa hoàn toàn ngẫu nhiên như trường hợp `MockEmbedder` ở đây — trong khi 4 câu còn lại (chỉ dựa vào similarity) đều trượt. Nói cách khác, với các câu hỏi mơ hồ về đối tượng, lọc metadata là **lớp bảo vệ đúng đắn duy nhất không thể thay thế bằng việc cải thiện mô hình nhúng**.

### Phân tích chung

Kết quả **1/5** này **chưa đánh giá được chất lượng chiến lược chunking**, vì vẫn chạy trên `MockEmbedder` (vector sinh từ md5, không mang ngữ nghĩa). Không gian tìm kiếm tăng lên **156 chunk** thì xác suất bốc trúng ngẫu nhiên giảm xuống. Câu 4 là ví dụ rõ nhất: chunk có **điểm cao nhất toàn bộ benchmark (+0.4706)** lại là đoạn nói về hoàn phí vận chuyển, chẳng liên quan gì đến câu hỏi về chế tài; điểm số cao ở đây phản ánh may rủi của hàm băm chứ không phản ánh độ liên quan.

Giữ nguyên bảng này làm **baseline sàn**: sau khi cài backend nhúng thật (`pip install -r requirements-local.txt`), mọi con số vượt trên 1/5 mới là cải thiện thật sự đến từ chiến lược chunking.

**Hạn chế đã phát hiện và cách khắc phục:** ban đầu `KnowledgeBaseAgent.answer` chỉ gọi `store.search`, nên ở câu 5 phần lọc mới chỉ kiểm chứng được ở tầng `EmbeddingStore`, còn agent vẫn nhận ngữ cảnh chưa lọc. Bổ sung tham số `metadata_filter` vào `answer` (tách helper `_retrieve`, tự chọn `search_with_filter` khi có filter, giữ nguyên chữ ký cũ nên không phá vỡ    test nào). Kết quả chạy lại câu 5 **xuyên suốt tới câu trả lời cuối cùng** nằm ở mục dưới đây.

### Chạy lại với mô hình nhúng THẬT (LocalEmbedder)

Chạy lại **đúng 5 câu hỏi đó**, chỉ thay backend nhúng:

```
python -m src.bench --local
```


> **Thiết lập:** `RecursiveChunker(chunk_size=700)` → **150 chunk**, cùng backend `paraphrase-multilingual-MiniLM-L12-v2`, cùng 5 câu hỏi, `top_k=3`.

| doc_id | `recursive` | audience |
|--------|------------|----------|
| `shopee-dispute-resolution` | 10 | buyer |
| `shopee-listing-policy` | 36 | seller |
| `shopee-prohibited-items` | 22 | seller |
| `shopee-return-policy` | 40 | buyer |
| `shopee-shipping-policy` | 42 | seller |
| **Tổng** | **150** | |

**Kết quả — `RecursiveChunker` đạt 5/5:**

| # | Dạng hỏi | Top-1 Chunk truy xuất được | Score | Hit@3 |
|---|----------|---------------------------|-------|-------|
| 1 | Tra số liệu | `shopee-return-policy` — chunk chứa mốc 15 ngày | **+0.8232** | ✅ |
| 2 | Hỏi điều kiện | `shopee-return-policy` — chunk liệt kê điều kiện trả hàng | **+0.7542** | ✅ |
| 3 | Hỏi quy trình | `shopee-dispute-resolution` | **+0.7654** | ✅ |
| 4 | Liệt kê | `shopee-prohibited-items` — "Việc áp dụng các biện pháp chế tài..." | **+0.7719** | ✅ |
| 5 | Cần filter (**có lọc**) | `shopee-return-policy` (+0.7100); chunk gold ở **hạng #2** (+0.7021) | | ✅ |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5**


**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Cần chú ý đến mật độ của văn bản trước khi chọn chiến lược chunking. Nên test thêm nhiều thiết lập với các backend trả phí khác*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5/ 5 |
| Hướng tiếp cận của tôi (My Approach) | 10/ 10 |
| Hoàn thiện code (Core Implementation — tests) | 30/ 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5/ 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10/ 10 |
| **Tổng phần cá nhân** | **60/ 60** |
