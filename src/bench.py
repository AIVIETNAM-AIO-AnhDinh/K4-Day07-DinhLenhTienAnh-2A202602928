#!/usr/bin/env python3
"""Benchmark truy xuất trên bộ tài liệu Shopee (Bài tập 3.x — Nhóm).

Chạy (từ thư mục gốc của repo):
    python -m src.bench                          # MockEmbedder (baseline sàn)
    python -m src.bench --local                  # LocalEmbedder (đa ngôn ngữ)
    python -m src.bench --local --chunker markdown   # đổi chiến lược chia nhỏ

In ra: số chunk đã nạp + kết quả top-3 cho cả 5 câu hỏi đánh giá.
Câu 5 chạy hai lần (không lọc / có metadata_filter) để cho thấy vì sao
lọc theo `audience` là bắt buộc với câu hỏi không nêu rõ đối tượng.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src import (
    Document,
    EmbeddingStore,
    KnowledgeBaseAgent,
    LocalEmbedder,
    MarkdownSectionChunker,
    RecursiveChunker,
    _mock_embed,
)

CORPUS_DIR = Path("data/ecommerce")
CHUNK_SIZE = 700
TOP_K = 3

CHUNKERS = {
    "recursive": lambda: RecursiveChunker(chunk_size=CHUNK_SIZE),
    "markdown": lambda: MarkdownSectionChunker(chunk_size=CHUNK_SIZE),
}
DEFAULT_CHUNKER = "recursive"

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# --------------------------------------------------------------------------
# 5 câu hỏi đánh giá. Mỗi câu có gold answer TRÍCH NGUYÊN VĂN từ tài liệu,
# không suy đoán chính sách của Shopee.
# --------------------------------------------------------------------------
QUERIES = [
    {
        "no": 1,
        "type": "Tra số liệu",
        "question": "Người Mua có bao nhiêu ngày để gửi yêu cầu trả hàng/hoàn tiền kể từ khi đơn hàng được cập nhật giao hàng thành công?",
        "gold": "15 (mười lăm) ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công. "
                "Riêng thực phẩm tươi sống và đông lạnh: trong vòng 24 giờ.",
        "gold_doc": "shopee-return-policy",
        "gold_ref": "Mục 3.2",
        "filter": None,
    },
    {
        "no": 2,
        "type": "Hỏi điều kiện",
        "question": "Người Mua được quyền yêu cầu trả hàng/hoàn tiền trong những trường hợp nào?",
        "gold": "Không nhận được Sản Phẩm / không nhận đủ / nhận hàng giả, hàng nhái; Sản Phẩm bị lỗi "
                "hoặc hư hại khi vận chuyển; Người Bán giao sai Sản Phẩm; Sản Phẩm khác biệt rõ rệt so "
                "với mô tả; Sản Phẩm hết hạn sử dụng; Người Bán đã tự thỏa thuận đồng ý cho trả hàng; "
                "Trả hàng COM (nguyên vẹn, không còn nhu cầu).",
        "gold_doc": "shopee-return-policy",
        "gold_ref": "Mục 3.1",
        "filter": None,
    },
    {
        "no": 3,
        "type": "Hỏi quy trình",
        "question": "Quy trình giải quyết tranh chấp/khiếu nại của Shopee gồm những bước nào?",
        "gold": "4 bước. B1: Người Mua bấm khiếu nại trong mục 'Đơn Mua' trên app/website. "
                "B2: Bộ phận giải quyết khiếu nại tiếp nhận yêu cầu. "
                "B3: Khiếu nại Trả Hàng/Hoàn Tiền xử lý theo Chính Sách Trả Hàng Và Hoàn Tiền; tranh chấp "
                "khác được đưa hướng giải quyết trong vòng 07 ngày làm việc. "
                "B4: Ngoài thẩm quyền của Sàn thì đưa ra cơ quan nhà nước có thẩm quyền.",
        "gold_doc": "shopee-dispute-resolution",
        "gold_ref": "Mục 1, Bước 1-4",
        "filter": None,
    },
    {
        "no": 4,
        "type": "Liệt kê",
        "question": "Người Bán vi phạm Chính Sách Cấm/Hạn Chế Sản Phẩm có thể bị áp dụng những chế tài nào?",
        "gold": "(i) Sản phẩm bị xóa; (ii) Tài khoản bị giới hạn quyền; (iii) Tài khoản bị đình chỉ hoạt "
                "động hoặc bị xóa; (iv) Cấn trừ số dư tài khoản Shopee, phong tỏa quyền rút tiền; "
                "(v) Các chế tài khác gồm phạt hành chính, xử lý hình sự và/hoặc bồi thường thiệt hại.",
        "gold_doc": "shopee-prohibited-items",
        "gold_ref": "Mục 3",
        "filter": None,
    },
    {
        # Câu bẫy: KHÔNG nêu rõ người hỏi là Người Mua hay Người Bán, trong khi
        # corpus có hai tài liệu cùng nói về "thời gian xử lý khiếu nại",
        # dùng chung từ vựng ("khiếu nại", "ngày làm việc"), nhưng khác audience
        # VÀ khác đáp án. Không lọc -> agent rất dễ trả lời sai đối tượng.
        "no": 5,
        "type": "Tra số liệu (CẦN metadata_filter)",
        "question": "Shopee xử lý khiếu nại trong bao nhiêu ngày làm việc?",
        "gold": "Với Người Mua (tranh chấp không phải khiếu nại Trả Hàng/Hoàn Tiền): trong vòng "
                "07 ngày làm việc kể từ ngày nhận đủ thông tin/tài liệu.",
        "gold_doc": "shopee-dispute-resolution",
        "gold_ref": "Mục 1, Bước 3",
        "filter": {"audience": "buyer"},
        "counterpart": (
            "shopee-shipping-policy (audience=seller): khiếu nại vận chuyển được xử lý "
            "TỐI ĐA 10 NGÀY LÀM VIỆC — cùng từ vựng, khác đối tượng, khác đáp án."
        ),
    },
]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tách YAML frontmatter đơn giản (key: "value") ra khỏi phần thân."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text

    metadata: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        value = value.split("#")[0].strip().strip('"').strip("'")
        if value:
            metadata[key.strip()] = value
    return metadata, text[match.end():]


def load_corpus(chunker_name: str = DEFAULT_CHUNKER) -> list[Document]:
    chunker = CHUNKERS[chunker_name]()
    docs: list[Document] = []

    for path in sorted(CORPUS_DIR.glob("shopee-*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata, _ = parse_frontmatter(raw)
        doc_id = metadata.get("doc_id", path.stem)

        for index, chunk in enumerate(chunker.chunk(raw)):
            docs.append(
                Document(
                    id=doc_id,
                    content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "source": str(path),
                        "audience": metadata.get("audience", "unknown"),
                        "category": metadata.get("category", "unknown"),
                        "source_url": metadata.get("source_url", ""),
                        "retrieved_at": metadata.get("retrieved_at", ""),
                        "chunk_no": index,
                    },
                )
            )
    return docs


def preview(text: str, width: int = 96) -> str:
    flat = " ".join(text.split())
    return flat[:width] + ("..." if len(flat) > width else "")


def show_results(results: list[dict], gold_doc: str) -> bool:
    if not results:
        print("      (không có kết quả)")
        return False

    hit = False
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        match = meta["doc_id"] == gold_doc
        hit = hit or match
        flag = "✅" if match else "  "
        print(f"      {flag} #{rank} score={result['score']:+.4f}  [{meta['doc_id']} | audience={meta['audience']}]")
        print(f"            {preview(result['content'])}")
    return hit


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark truy xuất Shopee")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Dùng LocalEmbedder (sentence-transformers) thay cho MockEmbedder",
    )
    parser.add_argument(
        "--chunker",
        choices=sorted(CHUNKERS),
        default=DEFAULT_CHUNKER,
        help=f"Chiến lược chia nhỏ (mặc định: {DEFAULT_CHUNKER})",
    )
    args = parser.parse_args()

    if args.local:
        print("Đang tải mô hình nhúng cục bộ (lần đầu sẽ mất thời gian tải về)...")
        embedder = LocalEmbedder()
    else:
        embedder = _mock_embed

    docs = load_corpus(args.chunker)
    store = EmbeddingStore(collection_name="shopee_bench", embedding_fn=embedder)
    store.add_documents(docs)

    by_doc: dict[str, int] = {}
    for doc in docs:
        by_doc[doc.metadata["doc_id"]] = by_doc.get(doc.metadata["doc_id"], 0) + 1

    print("=" * 100)
    print("BENCHMARK TRUY XUẤT — BỘ TÀI LIỆU SHOPEE")
    print("=" * 100)
    print(f"Chunker     : {CHUNKERS[args.chunker]().__class__.__name__}(chunk_size={CHUNK_SIZE})")
    print(f"Embeddings  : {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"Tài liệu    : {len(by_doc)}")
    print(f"TỔNG SỐ CHUNK ĐÃ NẠP: {store.get_collection_size()}")
    for doc_id, count in sorted(by_doc.items()):
        audience = next(d.metadata["audience"] for d in docs if d.metadata["doc_id"] == doc_id)
        print(f"   - {doc_id:<32} {count:>3} chunk   (audience={audience})")

    agent = KnowledgeBaseAgent(store, llm_fn=lambda p: f"[DEMO LLM] tổng hợp từ {p.count('(source:')} chunk")

    hits_top3 = 0
    for item in QUERIES:
        print()
        print("-" * 100)
        print(f"CÂU {item['no']} [{item['type']}]")
        print(f"  Hỏi        : {item['question']}")
        print(f"  Gold answer: {item['gold']}")
        print(f"  Nguồn      : {item['gold_doc']} ({item['gold_ref']})")

        if item["filter"] is None:
            print(f"  TOP-{TOP_K} (không lọc):")
            hit = show_results(store.search(item["question"], top_k=TOP_K), item["gold_doc"])
        else:
            print(f"  ⚠️  Câu hỏi KHÔNG nêu rõ đối tượng. Tài liệu đối chứng:")
            print(f"      {item['counterpart']}")
            print(f"  TOP-{TOP_K} KHÔNG lọc  → dễ lẫn hai đối tượng:")
            unfiltered = store.search(item["question"], top_k=TOP_K)
            show_results(unfiltered, item["gold_doc"])
            audiences = {r["metadata"]["audience"] for r in unfiltered}
            print(f"      → audience trong kết quả: {sorted(audiences)}"
                  f"{'  ❌ LẪN ĐỐI TƯỢNG' if len(audiences) > 1 else ''}")

            print(f"  TOP-{TOP_K} CÓ lọc metadata_filter={item['filter']}:")
            filtered = store.search_with_filter(item["question"], top_k=TOP_K, metadata_filter=item["filter"])
            hit = show_results(filtered, item["gold_doc"])
            audiences = {r["metadata"]["audience"] for r in filtered}
            print(f"      → audience trong kết quả: {sorted(audiences)}"
                  f"{'  ✅ ĐÚNG ĐỐI TƯỢNG' if audiences == {'buyer'} else ''}")

        hits_top3 += hit
        print(f"  Hit@{TOP_K}     : {'✅ CÓ' if hit else '❌ KHÔNG'}")
        print(f"  Agent      : "
              f"{agent.answer(item['question'], top_k=TOP_K, metadata_filter=item['filter'])}")

    print()
    print("=" * 100)
    print(f"KẾT QUẢ: {hits_top3}/{len(QUERIES)} câu có chunk đúng tài liệu trong top-{TOP_K}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
