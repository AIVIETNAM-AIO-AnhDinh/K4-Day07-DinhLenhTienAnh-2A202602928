from __future__ import annotations

from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    SYSTEM_INSTRUCTION = (
        "You are a knowledge base assistant. Answer the question using ONLY the "
        "context below. If the context does not contain the answer, say that you "
        "do not know instead of guessing. Cite the context number you relied on."
    )

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def _retrieve(self, question: str, top_k: int, metadata_filter: dict | None) -> list[dict]:
        """Truy xuất chunk, có lọc metadata nếu được yêu cầu.

        Lọc trước khi xếp hạng (pre-filtering) nên top_k luôn được lấy đủ
        trong phạm vi hợp lệ, thay vì bị cắt còn ít hơn k sau khi lọc.
        """
        if metadata_filter:
            return self.store.search_with_filter(
                question, top_k=top_k, metadata_filter=metadata_filter
            )
        return self.store.search(question, top_k=top_k)

    def _build_context(self, results: list[dict]) -> str:
        if not results:
            return "(No relevant context was found in the knowledge base.)"

        blocks = []
        for index, result in enumerate(results, start=1):
            source = result.get("metadata", {}).get("source", result.get("id", "unknown"))
            blocks.append(f"[{index}] (source: {source})\n{result['content']}")
        return "\n\n".join(blocks)

    def answer(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> str:
        """Trả lời câu hỏi theo mẫu RAG.

        metadata_filter: giới hạn truy xuất trong phạm vi metadata cho trước
            (ví dụ ``{"audience": "buyer"}``). Bắt buộc dùng với các câu hỏi
            không nêu rõ đối tượng, khi corpus chứa nhiều tài liệu cùng chủ đề
            nhưng khác đối tượng và khác đáp án.
        """
        results = self._retrieve(question, top_k, metadata_filter)
        prompt = (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            f"=== CONTEXT ===\n{self._build_context(results)}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"=== ANSWER ===\n"
        )
        return str(self.llm_fn(prompt))
