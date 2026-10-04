"""嵌入模型：一个**确定性的本地玩具实现** + 一个真实接口。

## 为什么默认用玩具实现

RAG 的评测如果依赖真实 embedding API，就会：每次数字可能不同、要花钱、
可能因为限流跑不完、而且**你分不清"检索变好"是因为换了切分策略还是因为模型服务抖了**。

所以我用哈希技巧做一个确定性的本地嵌入：把 token 哈希到固定维度的桶里，
按词频加权，再 L2 归一化。它**没有任何语义理解能力**，但它有两个关键性质：

  * **确定性**：同样的输入永远得到同样的向量，换个进程也一样（用 md5 而不是内置 hash，
    因为 Python 的 `hash()` 有随机盐）；
  * **字面相似**：共享 token 越多的文本，余弦相似度越高。

这刚好模拟了"纯词面相似度"的检索基线。真实语义检索的增益，
靠 `--live` 模式换真实 embedding 来测 —— **两者对比本身就是一个有价值的实验**。

## 中文为什么要用字符二元组

中文没有空格分词。用单字，区分度太低（"的""是""在"到处都是）；
接一个分词库又会引入依赖。字符二元组（bigram）是零依赖下的最佳折中：
"检索增强" → 检索、索增、增强。它天然覆盖了大部分双字词。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from typing import Sequence

__all__ = ["Embedder", "HashingEmbedder", "OpenAICompatEmbedder", "tokenize", "cosine"]

_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

# 中文检索里几乎没有区分度的高频字，去掉能明显提升字面相似度的信号质量
_STOP_CHARS = set("的了是在和与及或也就都而及其这那我你他她它们个中上下对为以于")

# 英文停用词：这些词在检索里只贡献噪声。注意 "is" 长度是 2，
# 如果只按长度过滤会被保留下来（长度过滤只挡单字母）。
_STOP_WORDS = frozenset("""
a an the and or but if then so as of to in on at for with from by
is are was were be been being am do does did doing have has had having
it its this that these those there here i you he she they we me him her them us
my your his their our mine yours theirs ours
will would shall should can could may might must
not no nor only own same than too very just also
what which who whom whose when where why how
all any both each few more most other some such
about above after again against between into over under up down out off
""".split())


def tokenize(text: str) -> list[str]:
    """零依赖中英混合分词：英文按词（去停用词），中文按单字 + 二元组。"""
    if not text:
        return []
    low = text.lower()
    tokens: list[str] = [w for w in _WORD_RE.findall(low)
                         if len(w) > 1 and w not in _STOP_WORDS]
    for run in _CJK_RE.findall(low):
        cleaned = "".join(ch for ch in run if ch not in _STOP_CHARS)
        tokens.extend(cleaned)                                     # 单字
        tokens.extend(cleaned[i:i + 2] for i in range(len(cleaned) - 1))  # 二元组
    return tokens


def _bucket(token: str, dim: int) -> int:
    """用 md5 而不是内置 hash：内置 hash 有随机盐，跨进程不可复现。"""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


class Embedder:
    """接口：只要能把文本变成等长向量就能接进检索。"""

    dim: int = 0
    name: str = "base"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """确定性哈希嵌入。零依赖、离线、可复现；**不具备语义理解能力**。"""

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim < 8:
            raise ValueError("dim 太小")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts = Counter(tokenize(text))
        if not counts:
            return vector
        for token, count in counts.items():
            # 次线性词频加权：出现 10 次不该等于出现 1 次的 10 倍重要
            weight = 1.0 + math.log(count)
            vector[_bucket(token, self.dim)] += weight
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。两边都已归一化时就是点积。"""
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class OpenAICompatEmbedder(Embedder):
    """真实嵌入接口（OpenAI 兼容），仅用标准库。用于 `--live` 对比实验。"""

    def __init__(self, api_key: str, *, model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1", timeout: float = 60.0) -> None:
        if not api_key:
            raise ValueError("api_key 为空")
        self.api_key = api_key
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.dim = 0  # 第一次调用后才知道

    @classmethod
    def from_env(cls) -> "OpenAICompatEmbedder":
        key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("没有找到 EMBEDDING_API_KEY / OPENAI_API_KEY")
        return cls(key, base_url=os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
                   model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.name, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        vectors = [item["embedding"] for item in payload["data"]]
        if vectors:
            self.dim = len(vectors[0])
        return vectors
