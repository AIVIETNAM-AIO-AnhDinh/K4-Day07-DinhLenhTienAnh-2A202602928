#!/usr/bin/env python3
"""A/B test: chạy câu hỏi CẦN metadata_filter hai lần (có lọc / không lọc)
trên cả ba chiến lược chia nhỏ, để trả lời câu "metadata filter có giúp ích không?"
(REPORT_NHOM mục 3).

Chạy (từ thư mục gốc repo):
    python -m src.ab_filter              # MockEmbedder
    python -m src.ab_filter --local      # LocalEmbedder (khuyến nghị)

Nếu top-3 của hai lần chạy GIỐNG HỆT NHAU trên mọi chiến lược thì câu hỏi
chưa thực sự cần filter — script sẽ báo FAIL để biết mà sửa câu hỏi/tách tài liệu.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    LocalEmbedder,
    MarkdownSectionChunker,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)
from src.bench import CORPUS_DIR, parse_frontmatter, preview

TOP_K = 3

# Câu hỏi KHÔNG nêu rõ đối tượng; corpus có hai tài liệu cùng chủ đề,
# cùng từ vựng ("khiếu nại", "ngày làm việc"), khác audience, khác đáp án.
QUESTION = "Shopee xử lý khiếu nại trong bao nhiêu ngày làm việc?"
METADATA_FILTER = {"audience": "buyer"}
GOLD_DOC = "shopee-dispute-resolution"     # audience=buyer  -> 07 ngày làm việc
TRAP_DOC = "shopee-shipping-policy"        # audience=seller -> tối đa 10 ngày làm việc

STRATEGIES = {
    "fixed_size": FixedSizeChunker(chunk_size=700, overlap=70),
    "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
    "recursive": RecursiveChunker(chunk_size=700),
    "markdown_section (custom)": MarkdownSectionChunker(chunk_size=700),
}


def load_docs(chunker) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("shopee-*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(raw)
        doc_id = meta.get("doc_id", path.stem)
        for index, chunk in enumerate(chunker.chunk(raw)):
            docs.append(
                Document(
                    id=doc_id,
                    content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "audience": meta.get("audience", "unknown"),
                        "category": meta.get("category", "unknown"),
                        "chunk_no": index,
                    },
                )
            )
    return docs


def show(results: list[dict]) -> tuple[list[str], set[str], bool]:
    """In top-3, trả về (danh sách doc_id, tập audience, có trúng gold không)."""
    if not results:
        print("        (rỗng)")
        return [], set(), False

    doc_ids, audiences, hit = [], set(), False
    for rank, r in enumerate(results, start=1):
        meta = r["metadata"]
        doc_ids.append(meta["doc_id"])
        audiences.add(meta["audience"])
        mark = "✅" if meta["doc_id"] == GOLD_DOC else ("⚠️ " if meta["doc_id"] == TRAP_DOC else "  ")
        hit = hit or meta["doc_id"] == GOLD_DOC
        print(f"        {mark} #{rank} {r['score']:+.4f} [{meta['doc_id']} | {meta['audience']}]")
        print(f"              {preview(r['content'], 88)}")
    return doc_ids, audiences, hit


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B metadata_filter trên 3 chiến lược chunking")
    parser.add_argument("--local", action="store_true", help="Dùng LocalEmbedder")
    args = parser.parse_args()

    embedder = LocalEmbedder() if args.local else _mock_embed

    print("=" * 100)
    print("A/B TEST — metadata_filter CÓ GIÚP ÍCH KHÔNG?")
    print("=" * 100)
    print(f"Câu hỏi   : {QUESTION}")
    print(f"Filter    : {METADATA_FILTER}")
    print(f"Gold doc  : {GOLD_DOC}   (audience=buyer  -> 07 ngày làm việc)")
    print(f"Trap doc  : {TRAP_DOC}   (audience=seller -> tối đa 10 ngày làm việc)")
    print(f"Embeddings: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    summary = []
    for name, chunker in STRATEGIES.items():
        docs = load_docs(chunker)
        store = EmbeddingStore(collection_name=f"ab_{name}", embedding_fn=embedder)
        store.add_documents(docs)

        print()
        print("-" * 100)
        print(f"CHIẾN LƯỢC: {name}   ({store.get_collection_size()} chunk)")

        print(f"  [A] KHÔNG lọc:")
        a_ids, a_aud, a_hit = show(store.search(QUESTION, top_k=TOP_K))
        print(f"      audience = {sorted(a_aud)}"
              f"{'   ❌ LẪN ĐỐI TƯỢNG' if len(a_aud) > 1 else ''}")

        print(f"  [B] CÓ lọc metadata_filter={METADATA_FILTER}:")
        b_ids, b_aud, b_hit = show(
            store.search_with_filter(QUESTION, top_k=TOP_K, metadata_filter=METADATA_FILTER)
        )
        print(f"      audience = {sorted(b_aud)}"
              f"{'   ✅ ĐÚNG ĐỐI TƯỢNG' if b_aud == {'buyer'} else ''}")

        identical = a_ids == b_ids
        print(f"  => Top-3 hai lần {'GIỐNG HỆT ❌' if identical else 'KHÁC NHAU ✅'}"
              f" | trap doc trong [A]: {'CÓ ⚠️' if TRAP_DOC in a_ids else 'không'}"
              f" | gold: [A]={'✅' if a_hit else '❌'} [B]={'✅' if b_hit else '❌'}")
        summary.append((name, store.get_collection_size(), a_ids, b_ids, a_aud, b_aud, a_hit, b_hit, identical))

    print()
    print("=" * 100)
    print("TỔNG HỢP")
    print("=" * 100)
    print(f"{'Chiến lược':<28}{'#chunk':>7}  {'[A] audience':<18}{'[B] audience':<14}"
          f"{'gold A':<8}{'gold B':<8}{'A≠B':<6}")
    for name, n, a_ids, b_ids, a_aud, b_aud, a_hit, b_hit, identical in summary:
        print(f"{name:<28}{n:>7}  {','.join(sorted(a_aud)):<18}{','.join(sorted(b_aud)):<14}"
              f"{'✅' if a_hit else '❌':<8}{'✅' if b_hit else '❌':<8}{'✅' if not identical else '❌':<6}")

    all_differ = all(not item[-1] for item in summary)
    print()
    if all_differ:
        print("KẾT LUẬN: ✅ Trên MỌI chiến lược, top-3 có lọc KHÁC top-3 không lọc")
        print("          => câu hỏi THỰC SỰ cần metadata_filter.")
    else:
        same = [item[0] for item in summary if item[-1]]
        print(f"KẾT LUẬN: ❌ Top-3 giống hệt nhau ở: {', '.join(same)}")
        print("          => cần sửa lại câu hỏi hoặc tách tài liệu theo audience.")
    return 0 if all_differ else 1


if __name__ == "__main__":
    raise SystemExit(main())
