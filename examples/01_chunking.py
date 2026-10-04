#!/usr/bin/env python3
"""示例 1：三种切分策略的直观对比。

运行：python examples/01_chunking.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from minirag import DOCUMENTS, chunk_document, chunk_stats  # noqa: E402

LINE = "=" * 88


def main() -> None:
    document = next(d for d in DOCUMENTS if d.id == "doc-rag")
    print(f"\n{LINE}\n原文《{document.title}》共 {len(document.text)} 字\n{LINE}")
    print(document.text[:120].replace("\n", " ") + "……")

    for strategy in ("fixed", "recursive", "heading"):
        chunks = chunk_document(document.id, document.title, document.text,
                                strategy=strategy, size=180, overlap=30)
        stats = chunk_stats(chunks)
        print(f"\n{'-' * 88}")
        print(f"策略 `{strategy}`：{stats['count']} 块，"
              f"平均 {stats['mean_chars']} 字，范围 {stats['min_chars']}–{stats['max_chars']}")
        print(f"{'-' * 88}")
        for chunk in chunks[:3]:
            head = f"【{chunk.heading}】" if chunk.heading else ""
            preview = chunk.text.replace("\n", " ")[:66]
            print(f"  [{chunk.index}] {head}{preview}…")
        if len(chunks) > 3:
            print(f"  …（共 {len(chunks)} 块）")

    print(f"\n{LINE}\n怎么选\n{LINE}")
    print("  * `fixed` 最省事，块长可控，但会把句子切断 —— 上表里最小块只有 61 字就是被截断的证据。")
    print("  * `recursive` 是通用默认：优先在段落/句号处断开，块长不均匀但语义完整。")
    print("  * `heading` 利用文档结构，块最小（每块只装一个小节），")
    print("    好处是引用时能带上章节标题，坏处是块太碎、丢了上下文。")
    print("  * `overlap`（重叠）用于补偿边界问题：相邻块共享 30 字，")
    print("    代价是存储与召回冗余。这是唯一不需要重训模型就能拿到的收益。")

    print(f"\n  全文共 {len(DOCUMENTS)} 篇文档被切成 "
          f"{len(chunk_document('x', 't', DOCUMENTS[0].text, size=200))} 块（以首篇为例）。")


if __name__ == "__main__":
    main()
