"""生成层：把检索结果拼成带引用的答案。

两个工程要点：

1. **上下文预算**。检索返回的块不能全塞进提示词 —— 那是成本失控的头号原因。
   这里按 `max_context_chars` 截断，并优先保留排名靠前的块。
2. **引用可追溯**。给每个块编号，要求模型在答案里标注 `[1]` `[2]`，
   然后**回查编号是否真的来自检索结果**。这一步能把"模型编造出处"抓出来，
   也是 `citation_accuracy` 这个指标的来源。

离线模式用一个**抽取式假模型**：它从给定上下文里挑句子拼成答案并带引用。
这当然不是真回答，但足以让整条链路（检索 → 上下文 → 引用 → 校验）可端到端测试，
而且结果完全可复现。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Sequence

from .retrieval import Retriever, SearchHit

__all__ = ["Citation", "RagAnswer", "BaseLLM", "ExtractiveMockLLM", "OpenAICompatLLM",
           "RagPipeline", "citation_accuracy", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只依据下面提供的资料回答，不要使用资料之外的知识。\n"
    "每条结论后面必须标注来源编号，格式为 [1]、[2]。\n"
    "如果资料不足以回答，直接说「根据现有资料无法回答」，不要编造。"
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    heading: str = ""
    preview: str = ""

    def to_dict(self) -> dict:
        return {"index": self.index, "chunk_id": self.chunk_id, "doc_id": self.doc_id,
                "title": self.title, "heading": self.heading, "preview": self.preview}


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    cited_indexes: list[int] = field(default_factory=list)
    invalid_indexes: list[int] = field(default_factory=list)
    context_chars: int = 0
    retrieved: list[SearchHit] = field(default_factory=list)
    grounded: bool = False

    @property
    def citation_accuracy(self) -> float:
        """引用的编号里，有多少是真实来自检索结果的。"""
        if not self.cited_indexes:
            return 0.0
        good = [i for i in self.cited_indexes if i not in self.invalid_indexes]
        return len(good) / len(self.cited_indexes)

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
                "citations": [c.to_dict() for c in self.citations],
                "cited_indexes": self.cited_indexes, "invalid_indexes": self.invalid_indexes,
                "citation_accuracy": round(self.citation_accuracy, 4),
                "context_chars": self.context_chars, "grounded": self.grounded}


class BaseLLM:
    name = "base"

    def generate(self, *, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError


class ExtractiveMockLLM(BaseLLM):
    """离线假模型：从上下文里抽句子 + 自动带引用。确定性、零依赖。"""

    name = "extractive-mock"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences
        self.calls: list[str] = []

    def generate(self, *, system: str, user: str) -> str:
        self.calls.append(user)
        blocks = re.findall(r"\[(\d+)\]\s*(?:【([^】]*)】)?\s*(.*?)(?=\n\[\d+\]|\Z)",
                            user, re.DOTALL)
        if not blocks:
            return "根据现有资料无法回答。"

        pieces: list[str] = []
        for index, heading, body in blocks[: self.max_sentences]:
            sentences = [s.strip() for s in re.split(r"[。！？\n]", body) if len(s.strip()) > 12]
            if not sentences:
                continue
            head = f"【{heading}】" if heading else ""
            pieces.append(f"{head}{sentences[0]}。[{index}]")
        if not pieces:
            return "根据现有资料无法回答。"
        return "根据资料：" + " ".join(pieces)


class OpenAICompatLLM(BaseLLM):
    """真实模型（OpenAI 兼容），仅用标准库。用于 `--live`。"""

    def __init__(self, api_key: str, *, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1", timeout: float = 90.0,
                 temperature: float = 0.0) -> None:
        if not api_key:
            raise ValueError("api_key 为空")
        self.api_key = api_key
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> "OpenAICompatLLM":
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
        if not key:
            raise ValueError("没有找到 DEEPSEEK_API_KEY / LLM_API_KEY")
        return cls(key, model=os.environ.get("LLM_MODEL", "deepseek-chat"),
                   base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"))

    def generate(self, *, system: str, user: str) -> str:
        body = json.dumps({"model": self.name, "temperature": self.temperature,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (payload["choices"][0]["message"].get("content") or "").strip()


@dataclass
class RagPipeline:
    """检索 → 拼上下文 → 生成 → 引用校验。"""

    retriever: Retriever
    llm: BaseLLM
    top_k: int = 4
    mode: str = "hybrid"
    max_context_chars: int = 1200

    def build_context(self, hits: Sequence[SearchHit]) -> tuple[str, list[Citation]]:
        blocks: list[str] = []
        citations: list[Citation] = []
        used = 0
        for index, hit in enumerate(hits, start=1):
            heading = hit.chunk.heading
            header = f"[{index}] 【{heading or hit.chunk.title}】"
            piece = f"{header}\n{hit.chunk.text}"
            if used + len(piece) > self.max_context_chars and blocks:
                break                      # 预算用完就停，优先保留靠前的块
            blocks.append(piece)
            used += len(piece)
            citations.append(Citation(index=index, chunk_id=hit.chunk.id, doc_id=hit.doc_id,
                                      title=hit.chunk.title, heading=heading,
                                      preview=hit.chunk.text[:80]))
        return "\n\n".join(blocks), citations

    def answer(self, question: str) -> RagAnswer:
        hits = self.retriever.search(question, mode=self.mode, top_k=self.top_k)
        context, citations = self.build_context(hits)
        if not context:
            return RagAnswer(question=question, answer="根据现有资料无法回答。", retrieved=hits)

        user = f"资料：\n{context}\n\n问题：{question}\n\n请依据资料作答并标注引用编号。"
        text = self.llm.generate(system=SYSTEM_PROMPT, user=user)

        valid = {c.index for c in citations}
        cited = [int(n) for n in _CITATION_RE.findall(text)]
        invalid = sorted({n for n in cited if n not in valid})

        return RagAnswer(
            question=question, answer=text, citations=citations,
            cited_indexes=sorted(set(cited)), invalid_indexes=invalid,
            context_chars=len(context), retrieved=hits,
            grounded=bool(cited) and not invalid,
        )


def citation_accuracy(answers: Sequence[RagAnswer]) -> float:
    """所有答案的引用准确率均值（没有任何引用的答案记 0）。"""
    if not answers:
        return 0.0
    return sum(a.citation_accuracy for a in answers) / len(answers)
