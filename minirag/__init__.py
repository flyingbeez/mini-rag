"""minirag —— 从零实现的 RAG 检索链路，并**用自建评测集量化每一步的收益**。

    切分  chunk.py      固定长度 / 递归分隔符 / 标题感知，三种策略 + 重叠
    嵌入  embed.py      确定性哈希嵌入（零依赖、可复现）+ 真实 API 接口
    检索  retrieval.py  BM25（稀疏）+ 向量（稠密）+ RRF 融合
    生成  rag.py        上下文预算控制 + 引用可追溯 + 引用准确率
    评测  evaluate.py   Recall@k / Hit@k / MRR + 两个维度的消融实验
    语料  corpus.py     5 篇长文档 + 15 篇短文档 + 24 条带标注的 query
"""

from .chunk import Chunk, chunk_document, chunk_fixed, chunk_heading, chunk_recursive, chunk_stats
from .corpus import DOCUMENTS, QUERIES, Document, Query, documents_by_id, stats
from .embed import Embedder, HashingEmbedder, OpenAICompatEmbedder, cosine, tokenize
from .evaluate import (
    EvalReport,
    QueryResult,
    ablation_retrieval,
    build_chunks,
    build_retriever,
    evaluate,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    render_tables,
)
from .rag import (
    BaseLLM,
    Citation,
    ExtractiveMockLLM,
    OpenAICompatLLM,
    RagAnswer,
    RagPipeline,
    citation_accuracy,
)
from .retrieval import MODES, BM25, Retriever, SearchHit, VectorIndex, rrf_fuse

__version__ = "0.1.0"

__all__ = [
    # 切分
    "Chunk", "chunk_document", "chunk_fixed", "chunk_recursive", "chunk_heading", "chunk_stats",
    # 嵌入
    "Embedder", "HashingEmbedder", "OpenAICompatEmbedder", "tokenize", "cosine",
    # 检索
    "BM25", "VectorIndex", "Retriever", "SearchHit", "rrf_fuse", "MODES",
    # 生成
    "BaseLLM", "ExtractiveMockLLM", "OpenAICompatLLM", "RagPipeline", "RagAnswer",
    "Citation", "citation_accuracy",
    # 评测
    "evaluate", "ablation_retrieval", "build_chunks", "build_retriever",
    "recall_at_k", "hit_at_k", "reciprocal_rank", "render_tables",
    "EvalReport", "QueryResult",
    # 语料
    "DOCUMENTS", "QUERIES", "Document", "Query", "documents_by_id", "stats",
    "__version__",
]
