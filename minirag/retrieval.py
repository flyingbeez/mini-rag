"""检索：BM25（稀疏）+ 向量（稠密）+ RRF 融合。

三种模式的性格完全不同：

| 模式 | 擅长 | 不擅长 |
|---|---|---|
| `bm25` | 精确关键词、编号、专有名词（HNSW、RRF、max_tokens） | 同义改写（"记住上次聊的事" 搜不到 "长期记忆"） |
| `vector` | 语义相近、换了说法 | 专有名词、生僻词（哈希嵌入尤其如此） |
| `hybrid` | 两者覆盖的场景并集 | 需要调参、多一路开销 |

融合用 **RRF（倒数排名融合）**：`score(d) = Σ 1/(k + rank_i(d))`。
它的好处是**不需要归一化分数**——BM25 的分数量纲和余弦相似度毫无关系，
直接加权相加要调权重，而 RRF 只用排名，天然可比。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .chunk import Chunk
from .embed import Embedder, cosine, tokenize

__all__ = ["BM25", "VectorIndex", "Retriever", "SearchHit", "rrf_fuse", "MODES"]

MODES = ("vector", "bm25", "hybrid")


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

class BM25:
    """经典的 BM25（Okapi）实现，纯 Python。

    ``k1`` 控制词频饱和速度（词出现 10 次不该比 1 次重要 10 倍），
    ``b`` 控制文档长度归一化的强度。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 必须为正数")
        if not 0 <= b <= 1:
            raise ValueError("b 必须在 0~1 之间")
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = []
        self.doc_lengths: list[int] = []
        self.avg_length: float = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, documents: Sequence[str]) -> "BM25":
        self.doc_freqs = [Counter(tokenize(doc)) for doc in documents]
        self.doc_lengths = [sum(counter.values()) for counter in self.doc_freqs]
        self.avg_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0

        total = len(self.doc_freqs)
        seen: Counter = Counter()
        for counter in self.doc_freqs:
            seen.update(counter.keys())
        # 概率型 IDF：出现次数越多的词权重越低，且保证非负
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in seen.items()
        }
        return self

    def scores(self, query: str) -> list[float]:
        if not self.doc_freqs:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * len(self.doc_freqs)

        results = [0.0] * len(self.doc_freqs)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, counter in enumerate(self.doc_freqs):
                freq = counter.get(term, 0)
                if not freq:
                    continue
                length_norm = 1 - self.b + self.b * (self.doc_lengths[index] / (self.avg_length or 1))
                results[index] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * length_norm)
        return results


# --------------------------------------------------------------------------- #
# 向量索引
# --------------------------------------------------------------------------- #

class VectorIndex:
    """暴力余弦检索。语料只有几百个块，暴力足够快，也便于调试。"""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.vectors: list[list[float]] = []

    def fit_texts(self, texts: Sequence[str]) -> "VectorIndex":
        self.vectors = self.embedder.embed(list(texts)) if texts else []
        return self

    def scores(self, query: str) -> list[float]:
        if not self.vectors:
            return []
        query_vector = self.embedder.embed_one(query)
        return [cosine(query_vector, vector) for vector in self.vectors]


# --------------------------------------------------------------------------- #
# 融合
# --------------------------------------------------------------------------- #

def rrf_fuse(rankings: Sequence[Sequence[int]], *, k: int = 60,
             weights: Sequence[float] | None = None) -> dict[int, float]:
    """倒数排名融合。

    Args:
        rankings: 多路检索的结果，每路是一个**按相关性降序**的下标列表。
        k: RRF 平滑常数，常用 60。越大则各名次之间的差距越平缓。
        weights: 每一路的权重，默认都为 1。
    """
    if k <= 0:
        raise ValueError("k 必须为正数")
    fused: dict[int, float] = {}
    for path, ranking in enumerate(rankings):
        weight = weights[path] if weights else 1.0
        for position, doc_index in enumerate(ranking, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + weight / (k + position)
    return fused


def _ranking_from_scores(scores: Sequence[float], top: int | None = None) -> list[int]:
    """把分数转成降序下标列表（同分按原下标稳定排序）。"""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:top] if top else order


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #

@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int
    mode: str

    @property
    def doc_id(self) -> str:
        return self.chunk.source_id

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk.id, "doc_id": self.doc_id, "score": round(self.score, 6),
                "rank": self.rank, "mode": self.mode, "preview": self.chunk.text[:60]}


@dataclass
class Retriever:
    """把切分、BM25、向量索引打包在一起的检索器。"""

    chunks: list[Chunk]
    embedder: Embedder
    bm25: BM25 = field(default_factory=BM25)
    vector: VectorIndex | None = None

    def __post_init__(self) -> None:
        texts = [c.text for c in self.chunks]
        self.bm25.fit(texts)
        self.vector = VectorIndex(self.embedder).fit_texts(texts)

    # ------------------------------------------------------------------ #

    def search(self, query: str, *, mode: str = "hybrid", top_k: int = 5,
               candidate_k: int = 20, rrf_k: int = 60) -> list[SearchHit]:
        """检索。`candidate_k` 是两路各自召回的候选数，融合后再取 `top_k`。"""
        if mode not in MODES:
            raise ValueError(f"未知检索模式 {mode!r}，可选 {MODES}")
        if not self.chunks:
            return []

        if mode == "bm25":
            scores = self.bm25.scores(query)
            order = _ranking_from_scores(scores, top_k)
            return [SearchHit(self.chunks[i], scores[i], rank, mode)
                    for rank, i in enumerate(order, start=1)]

        if mode == "vector":
            scores = self.vector.scores(query) if self.vector else []
            order = _ranking_from_scores(scores, top_k)
            return [SearchHit(self.chunks[i], scores[i], rank, mode)
                    for rank, i in enumerate(order, start=1)]

        # hybrid：两路各取候选，RRF 融合
        bm25_scores = self.bm25.scores(query)
        vector_scores = self.vector.scores(query) if self.vector else []
        bm25_ranking = _ranking_from_scores(bm25_scores, candidate_k)
        vector_ranking = _ranking_from_scores(vector_scores, candidate_k)
        fused = rrf_fuse([bm25_ranking, vector_ranking], k=rrf_k)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return [SearchHit(self.chunks[i], fused[i], rank, mode)
                for rank, i in enumerate(order, start=1)]

    # ------------------------------------------------------------------ #

    def doc_ranking(self, query: str, *, mode: str = "hybrid", top_k: int = 10,
                    max_per_doc: int = 2) -> list[str]:
        """把块级结果聚合成文档级排名（检索系统通常在这个粒度上算指标）。

        `max_per_doc` 限制同一篇文档最多贡献几个块 —— 否则一篇长文档
        会用前几名把整个结果列表占满，其他相关文档根本没机会出现。
        """
        hits = self.search(query, mode=mode, top_k=top_k * max_per_doc)
        seen_docs: Counter = Counter()
        ranking: list[str] = []
        for hit in hits:
            if seen_docs[hit.doc_id] >= max_per_doc:
                continue
            seen_docs[hit.doc_id] += 1
            if hit.doc_id not in ranking:
                ranking.append(hit.doc_id)
            if len(ranking) >= top_k:
                break
        return ranking
