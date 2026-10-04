#!/usr/bin/env python3
"""示例 3：消融实验 —— 本仓库的核心产出。

运行：python examples/03_ablation.py
     python examples/03_ablation.py --live   # 换成真实 embedding 模型

离线模式用确定性哈希嵌入，所以任何人在任何机器上跑出的数字完全一致。
它测的是"检索链路与评测逻辑是否正确"，不是"真实语义检索的上限"。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from minirag import (  # noqa: E402
    DOCUMENTS,
    QUERIES,
    HashingEmbedder,
    ablation_retrieval,
    build_retriever,
    evaluate,
    render_tables,
    stats,
)
from minirag.retrieval import MODES  # noqa: E402

LINE = "=" * 92


def main() -> None:
    live = "--live" in sys.argv
    embedder = None
    if live:
        from minirag import OpenAICompatEmbedder
        embedder = OpenAICompatEmbedder.from_env()
        print(f"【真实嵌入模式】模型 = {embedder.name}")
    else:
        embedder = HashingEmbedder(dim=512)

    corpus_stats = stats()
    print(f"\n{LINE}")
    print("RAG 检索消融实验")
    print(f"语料：{corpus_stats['documents']} 篇文档"
          f"（长 {corpus_stats['long_documents']} / 短 {corpus_stats['short_documents']}），"
          f"共 {corpus_stats['total_chars']} 字")
    print(f"评测集：{corpus_stats['queries']} 条 query，"
          f"类型分布 {corpus_stats['query_kinds']}")
    print(f"嵌入：{embedder.name}（dim={embedder.dim}）")
    print(LINE)

    k = 5
    reports = ablation_retrieval(DOCUMENTS, QUERIES, k=k, embedder=embedder)

    print(f"\n{render_tables(reports, k=k)}")

    # ---- 失败案例分析 -------------------------------------------------- #
    print(f"\n{LINE}")
    print("失败案例分析（recursive + hybrid，前 8 条未命中）")
    print(LINE)
    retriever = build_retriever(DOCUMENTS, strategy="recursive", embedder=embedder)
    report = evaluate(retriever, QUERIES, mode="hybrid", k=k, label="recursive/hybrid")
    failures = report.failures()
    if not failures:
        print("  没有未命中的 query")
    for item in failures[:8]:
        query = next(q for q in QUERIES if q.qid == item.qid)
        print(f"  [{item.qid}] ({item.kind}) {query.text}")
        print(f"        期望 {list(item.relevant)}｜实际召回 {item.retrieved}")
        if query.note:
            print(f"        线索：{query.note}")

    # ---- 检索与问答端到端 ---------------------------------------------- #
    print(f"\n{LINE}")
    print("端到端：检索 + 带引用作答（离线提取式假模型）")
    print(LINE)
    from minirag import ExtractiveMockLLM, RagPipeline, citation_accuracy

    pipeline = RagPipeline(retriever=retriever, llm=ExtractiveMockLLM(), top_k=4)
    sample_questions = [QUERIES[0].text, QUERIES[7].text, QUERIES[13].text]
    answers = []
    for question in sample_questions:
        result = pipeline.answer(question)
        answers.append(result)
        print(f"\n问：{question}")
        print(f"答：{result.answer[:160]}")
        print(f"    引用 {result.cited_indexes}｜无效引用 {result.invalid_indexes}"
              f"｜上下文 {result.context_chars} 字")

    print(f"\n  引用准确率（全部 query）：{citation_accuracy(answers) * 100:.1f}%")
    print("  注意：无效引用会被抓出来 —— 这是「模型编造出处」的检测手段。")

    print(f"\n{LINE}")
    print("读表要点")
    print(LINE)
    print("  1. 看表 3 的按类型分解：keyword 类 BM25 占优，semantic 类向量占优，")
    print("     hybrid 在两类上都不吃亏 —— 这才是「混合检索值不值」的答案。")
    print("  2. 看表 2：切分策略改变分块数、平均块长，进而影响召回。")
    print("     这是唯一一个纯工程手段就能拿收益的环节。")
    print("  3. 看失败案例：每一种失败都对应一个具体的改进方向")
    print("     （补同义词、改切分粒度、加大候选数、上重排）。")


if __name__ == "__main__":
    main()
