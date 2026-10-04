"""切分策略 —— 三种做法，外加一个"重叠"参数。

为什么切分这么重要：它是 RAG 里**唯一一个你可以纯靠工程手段拿到收益**的环节。
换 embedding 模型要花钱、要重新索引；换切分策略只改几十行代码。

三种策略的取舍：

| 策略 | 优点 | 缺点 | 适用 |
|---|---|---|---|
| `fixed` 固定长度 | 实现最简单，块大小可控 | 会把句子/词从中间截断 | 无结构的长文本 |
| `recursive` 递归分隔符 | 尽量在语义边界断开 | 块大小不均匀 | 通用默认 |
| `heading` 标题感知 | 同章节内容在一起，可带来源标题 | 依赖文档有标题结构 | Markdown / 结构化文档 |

`overlap`（重叠）是通用的补偿手段：相邻块共享一部分文本，
避免答案正好落在块的边界上被切成两半。代价是存储与召回冗余。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Chunk", "chunk_fixed", "chunk_recursive", "chunk_heading", "chunk_document", "STRATEGIES"]


@dataclass
class Chunk:
    """一个切分单元。`source_id` 指回原文，用于引用溯源。"""

    id: str
    text: str
    source_id: str
    title: str = ""
    index: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def heading(self) -> str:
        """这个块所在的最近一级标题（如果有）。"""
        return self.meta.get("heading", "")

    def to_dict(self) -> dict:
        return {"id": self.id, "source_id": self.source_id, "title": self.title,
                "index": self.index, "chars": len(self.text), "heading": self.heading}


# --------------------------------------------------------------------------- #
# 1) 固定长度
# --------------------------------------------------------------------------- #

def chunk_fixed(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """按字符数硬切。实现最简单，也最容易把句子切断。"""
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")

    pieces: list[str] = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


# --------------------------------------------------------------------------- #
# 2) 递归分隔符
# --------------------------------------------------------------------------- #

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]


def chunk_recursive(text: str, size: int = 200, overlap: int = 40,
                    separators: list[str] | None = None) -> list[str]:
    """按分隔符优先级递归切分：先试段落，再试换行、句号……最后才硬切。"""
    if size <= 0:
        raise ValueError("size 必须为正数")
    seps = list(separators if separators is not None else _SEPARATORS)

    def _split(piece: str, depth: int) -> list[str]:
        if len(piece) <= size:
            return [piece] if piece.strip() else []
        if depth >= len(seps):
            return chunk_fixed(piece, size, max(0, min(overlap, size - 1)))

        sep = seps[depth]
        if not sep:
            return chunk_fixed(piece, size, max(0, min(overlap, size - 1)))

        parts = [p for p in piece.split(sep) if p.strip()]
        if len(parts) <= 1:
            return _split(piece, depth + 1)

        out: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{sep}{part}" if buffer else part
            if len(candidate) <= size:
                buffer = candidate
                continue
            if buffer:
                out.append(buffer)
            if len(part) > size:
                out.extend(_split(part, depth + 1))   # 单个片段还是太长，往下钻
                buffer = ""
            else:
                buffer = part
        if buffer:
            out.append(buffer)
        return out

    raw = _split(text, 0)
    return _apply_overlap(raw, overlap, size)


def _apply_overlap(pieces: list[str], overlap: int, size: int) -> list[str]:
    """给相邻块补上前一块的尾巴，避免答案落在边界上。"""
    if overlap <= 0 or len(pieces) <= 1:
        return [p.strip() for p in pieces if p.strip()]

    out: list[str] = []
    for index, piece in enumerate(pieces):
        if index == 0:
            out.append(piece.strip())
            continue
        tail = pieces[index - 1][-overlap:]
        merged = f"{tail}{piece}"
        out.append(merged[: size + overlap].strip())
    return [p for p in out if p]


# --------------------------------------------------------------------------- #
# 3) 标题感知
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_heading(text: str, size: int = 200, overlap: int = 40) -> list[tuple[str, str]]:
    """按 Markdown 标题切分，返回 (标题, 正文片段) 列表。

    没有标题结构的文档会退化成"整篇当一个章节"。
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
        elif heading:
            sections.append((heading, ""))

    out: list[tuple[str, str]] = []
    for heading, body in sections:
        if len(body) <= size:
            out.append((heading, body))
            continue
        for piece in chunk_recursive(body, size, overlap, separators=["\n\n", "\n", "。", " "]):
            out.append((heading, piece))
    return out


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #

STRATEGIES = ("fixed", "recursive", "heading")


def chunk_document(source_id: str, title: str, text: str, strategy: str = "recursive",
                   size: int = 200, overlap: int = 40) -> list[Chunk]:
    """把一篇文档切成 Chunk 列表。`strategy` 取 fixed / recursive / heading。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"未知切分策略 {strategy!r}，可选 {STRATEGIES}")

    chunks: list[Chunk] = []
    if strategy == "fixed":
        for index, piece in enumerate(chunk_fixed(text, size, overlap)):
            chunks.append(Chunk(id=f"{source_id}::{index}", text=piece,
                                source_id=source_id, title=title, index=index))
    elif strategy == "recursive":
        for index, piece in enumerate(chunk_recursive(text, size, overlap)):
            chunks.append(Chunk(id=f"{source_id}::{index}", text=piece,
                                source_id=source_id, title=title, index=index))
    else:
        for index, (heading, piece) in enumerate(chunk_heading(text, size, overlap)):
            chunks.append(Chunk(id=f"{source_id}::{index}", text=piece, source_id=source_id,
                                title=title, index=index, meta={"heading": heading}))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"count": 0, "mean_chars": 0, "min_chars": 0, "max_chars": 0}
    sizes = [len(c.text) for c in chunks]
    return {
        "count": len(chunks),
        "mean_chars": round(sum(sizes) / len(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
    }
