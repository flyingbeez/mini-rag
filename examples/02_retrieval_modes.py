#!/usr/bin/env python3
"""示例 2：三种检索模式在同一个问题上的实际表现。

运行：python examples/02_retrieval_modes.py

这个示例把"为什么需要混合检索"从口号变成可观察的现象。
（注意：本仓库默认用无语义能力的哈希嵌入，所以你会看到一个**反直觉的结果**，
详见 README 的说明。）
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from minirag import DOCUMENTS, HashingEmbedder, build_retriever  # noqa: E402

LINE = "=" * 88

QUESTIONS = [
    ("HNSW 索引的内存占用怎么样", "字面型：专有名词 HNSW 直接命中"),
    ("怎么让机器人记住上次聊过的事情", "语义型：字面说的是「长期记忆跨会话存在」"),
    ("熔断器半开状态放多少请求", "字面型：术语完全一致"),
]


def main() -> None:
    retriever = build_retriever(DOCUMENTS, strategy="recursive", embedder=HashingEmbedder(dim=512))

    for question, kind in QUESTIONS:
        print(f"\n{LINE}\n问：{question}\n（{kind}）\n{LINE}")
        for mode in ("bm25", "vector", "hybrid"):
            hits = retriever.search(question, mode=mode, top_k=3)
            print(f"\n  [mode={mode}]")
            for hit in hits:
                preview = hit.chunk.text.replace("\n", " ")[:52]
                print(f"    {hit.rank}. {hit.doc_id:<18} score={hit.score:8.5f}  {preview}…")
        print(f"\n  文档级排名（hybrid）：{retriever.doc_ranking(question, top_k=3)}")

    print(f"\n{LINE}\n读这个结果要注意\n{LINE}")
    print("  1. 本仓库的默认嵌入是**哈希字面相似度**，没有任何语义能力。")
    print("     所以它在语义型问题上并不比 BM25 强，混合检索也就没有互补信号。")
    print("  2. 换真实 embedding 模型后（`python examples/03_ablation.py --live`）")
    print("     向量路才真正提供独立信号，混合检索的价值才会显现。")
    print("  3. 这正是要做消融的原因：**混合检索不是无脑更好，")
    print("     它成立的前提是两路检索携带互相独立的信号。**")


if __name__ == "__main__":
    main()
