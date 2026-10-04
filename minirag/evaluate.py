"""评测：Recall@k / MRR + 两个维度的消融实验。

这是整个仓库最核心的部分。**调 RAG 没有评测集，就等于闭着眼睛改参数。**

两个消融维度：
  1. **检索模式** × 3（bm25 / vector / hybrid）—— 回答"混合检索到底值不值"
  2. **切分策略** × 3（fixed / recursive / heading）—— 回答"改切分值多少"

指标定义（都算在**文档粒度**上，因为这才是用户感知的"找没找到"）：
  * `Recall@k` —— 前 k 个文档里覆盖了多少比例的相关文档
  * `Hit@k`    —— 前 k 个里至少有一个相关文档的查询比例
  * `MRR`      —— 第一个相关文档排名的倒数均值，衡量"排得靠不靠前"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .chunk import Chunk, chunk_document, chunk_stats
from .corpus import DOCUMENTS, QUERIES, Document, Query
from .embed import Embedder, HashingEmbedder
from .retrieval import MODES, Retriever

__all__ = [
    "recall_at_k", "hit_at_k", "reciprocal_rank", "QueryResult", "EvalReport",
    "build_chunks", "build_retriever", "evaluate", "ablation_retrieval", "render_tables",
]


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / position
    return 0.0


@dataclass
class QueryResult:
    qid: str
    kind: str
    retrieved: list[str]
    relevant: tuple[str, ...]
    recall: float
    hit: float
    rr: float
    missed: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    label: str
    mode: str
    chunk_strategy: str
    k: int
    results: list[QueryResult] = field(default_factory=list)
    chunk_stats: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def recall(self) -> float:
        return sum(r.recall for r in self.results) / self.total if self.total else 0.0

    @property
    def hit(self) -> float:
        return sum(r.hit for r in self.results) / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return sum(r.rr for r in self.results) / self.total if self.total else 0.0

    def by_kind(self) -> dict[str, dict[str, float]]:
        """按 query 类型分解 —— 这是解释"为什么混合检索更好"的关键。"""
        buckets: dict[str, list[QueryResult]] = {}
        for result in self.results:
            buckets.setdefault(result.kind, []).append(result)
        out: dict[str, dict[str, float]] = {}
        for kind, items in buckets.items():
            n = len(items)
            out[kind] = {
                "n": n,
                "recall": round(sum(i.recall for i in items) / n, 4),
                "mrr": round(sum(i.rr for i in items) / n, 4),
            }
        return out

    def failures(self) -> list[QueryResult]:
        return [r for r in self.results if r.hit == 0.0]

    def to_dict(self) -> dict:
        return {"label": self.label, "mode": self.mode, "chunk_strategy": self.chunk_strategy,
                "k": self.k, "recall": round(self.recall, 4), "hit": round(self.hit, 4),
                "mrr": round(self.mrr, 4), "chunks": self.chunk_stats,
                "by_kind": self.by_kind()}


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #

def build_chunks(documents: Sequence[Document], strategy: str = "recursive",
                 size: int = 200, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document.id, document.title, document.text,
                                     strategy=strategy, size=size, overlap=overlap))
    return chunks


def build_retriever(documents: Sequence[Document] = DOCUMENTS, *, strategy: str = "recursive",
                    size: int = 200, overlap: int = 40,
                    embedder: Embedder | None = None) -> Retriever:
    chunks = build_chunks(documents, strategy=strategy, size=size, overlap=overlap)
    return Retriever(chunks=chunks, embedder=embedder or HashingEmbedder())


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #

def evaluate(retriever: Retriever, queries: Sequence[Query] = QUERIES, *, mode: str = "hybrid",
             k: int = 5, label: str = "", chunk_strategy: str = "recursive",
             with_failures: bool = True) -> EvalReport:
    if mode not in MODES:
        raise ValueError(f"未知检索模式 {mode!r}")
    report = EvalReport(label=label or f"{chunk_strategy}/{mode}", mode=mode,
                        chunk_strategy=chunk_strategy, k=k,
                        chunk_stats=chunk_stats(retriever.chunks))

    for query in queries:
        ranking = retriever.doc_ranking(query.text, mode=mode, top_k=k)
        result = QueryResult(
            qid=query.qid, kind=query.kind, retrieved=ranking, relevant=query.relevant,
            recall=recall_at_k(ranking, query.relevant, k),
            hit=hit_at_k(ranking, query.relevant, k),
            rr=reciprocal_rank(ranking, query.relevant),
        )
        if with_failures and result.hit == 0.0:
            result.missed = list(query.relevant)
        report.results.append(result)
    return report


def ablation_retrieval(documents: Sequence[Document] = DOCUMENTS,
                       queries: Sequence[Query] = QUERIES, *, k: int = 5,
                       strategies: Sequence[str] = ("fixed", "recursive", "heading"),
                       modes: Sequence[str] = MODES,
                       embedder: Embedder | None = None) -> dict[str, EvalReport]:
    """两个维度的完整消融：切分策略 × 检索模式。"""
    reports: dict[str, EvalReport] = {}
    for strategy in strategies:
        retriever = build_retriever(documents, strategy=strategy, embedder=embedder)
        for mode in modes:
            key = f"{strategy}/{mode}"
            reports[key] = evaluate(retriever, queries, mode=mode, k=k,
                                    label=key, chunk_strategy=strategy)
    return reports


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #

def render_tables(reports: dict[str, EvalReport], *, k: int = 5,
                  best_mode: str = "hybrid") -> str:
    lines: list[str] = []

    lines.append(f"### 表 1 · 检索模式消融（切分策略固定为 recursive，k={k}）\n")
    lines.append("| 检索模式 | Recall@" + str(k) + " | Hit@" + str(k) + " | MRR | 分块数 | 平均块长 |")
    lines.append("|---|---|---|---|---|---|")
    mode_rows = {m: reports[f"recursive/{m}"] for m in MODES if f"recursive/{m}" in reports}
    for mode, report in mode_rows.items():
        lines.append(f"| `{mode}` | {report.recall * 100:.1f}% | {report.hit * 100:.1f}% | "
                     f"{report.mrr:.3f} | {report.chunk_stats['count']} | "
                     f"{report.chunk_stats['mean_chars']} |")

    lines.append(f"\n### 表 2 · 切分策略消融（检索模式固定为 {best_mode}，k={k}）\n")
    lines.append("| 切分策略 | Recall@" + str(k) + " | Hit@" + str(k) + " | MRR | 分块数 | 平均块长 | 最小/最大 |")
    lines.append("|---|---|---|---|---|---|---|")
    strategy_rows = {key.split("/")[0]: report for key, report in reports.items()
                     if key.endswith(f"/{best_mode}")}
    for strategy, report in strategy_rows.items():
        stats = report.chunk_stats
        lines.append(f"| `{strategy}` | {report.recall * 100:.1f}% | {report.hit * 100:.1f}% | "
                     f"{report.mrr:.3f} | {stats['count']} | {stats['mean_chars']} | "
                     f"{stats['min_chars']}/{stats['max_chars']} |")

    lines.append(f"\n### 表 3 · 按 query 类型分解（recursive，k={k}）\n")
    lines.append("| 检索模式 | 类型 | 样本数 | Recall@" + str(k) + " | MRR |")
    lines.append("|---|---|---|---|---|")
    for mode, report in mode_rows.items():
        for kind, metrics in sorted(report.by_kind().items()):
            lines.append(f"| `{mode}` | {kind} | {metrics['n']} | "
                         f"{metrics['recall'] * 100:.1f}% | {metrics['mrr']:.3f} |")

    return "\n".join(lines)
