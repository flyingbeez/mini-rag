"""嵌入与检索测试。"""

from __future__ import annotations

import unittest

from minirag.chunk import chunk_document
from minirag.embed import HashingEmbedder, cosine, tokenize
from minirag.retrieval import BM25, Retriever, VectorIndex, _ranking_from_scores, rrf_fuse


class TestTokenize(unittest.TestCase):
    def test_english_words(self):
        tokens = tokenize("Python is great")
        self.assertIn("python", tokens)
        self.assertIn("great", tokens)
        self.assertNotIn("is", tokens, "英文停用词应被过滤")

    def test_single_letters_filtered(self):
        self.assertNotIn("a", tokenize("a b c python"))

    def test_chinese_bigrams(self):
        tokens = tokenize("检索增强")
        self.assertIn("检", tokens)
        self.assertIn("检索", tokens)
        self.assertIn("增强", tokens)

    def test_stop_chars_removed(self):
        self.assertNotIn("的", tokenize("好的的的"))

    def test_empty(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("，。！"), [])


class TestEmbedder(unittest.TestCase):
    def test_deterministic_across_instances(self):
        a = HashingEmbedder(dim=64).embed_one("检索增强生成")
        b = HashingEmbedder(dim=64).embed_one("检索增强生成")
        self.assertEqual(a, b, "必须跨实例、跨进程可复现（不能用内置 hash）")

    def test_normalised(self):
        vector = HashingEmbedder(dim=64).embed_one("hello 检索")
        norm = sum(v * v for v in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_dim(self):
        self.assertEqual(len(HashingEmbedder(dim=32).embed_one("x 检索")), 32)

    def test_similar_text_scores_higher(self):
        embedder = HashingEmbedder(dim=512)
        base = embedder.embed_one("BM25 是稀疏检索算法")
        near = embedder.embed_one("BM25 属于稀疏检索")
        far = embedder.embed_one("今天天气不错")
        self.assertGreater(cosine(base, near), cosine(base, far))

    def test_empty_text_is_zero_vector(self):
        vector = HashingEmbedder(dim=16).embed_one("，。！")
        self.assertEqual(sum(abs(v) for v in vector), 0.0)

    def test_cosine_edge_cases(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_cosine_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine([1.0], [1.0, 2.0])

    def test_invalid_dim(self):
        with self.assertRaises(ValueError):
            HashingEmbedder(dim=2)


class TestBM25(unittest.TestCase):
    DOCS = [
        "BM25 是稀疏检索算法，擅长精确关键词匹配",
        "向量检索擅长语义相似，但不懂专有名词",
        "混合检索把两路结果融合起来",
    ]

    def test_ranks_matching_doc_first(self):
        bm25 = BM25().fit(self.DOCS)
        scores = bm25.scores("BM25 稀疏检索")
        self.assertEqual(_ranking_from_scores(scores)[0], 0)

    def test_second_query(self):
        bm25 = BM25().fit(self.DOCS)
        scores = bm25.scores("向量 语义")
        self.assertEqual(_ranking_from_scores(scores)[0], 1)

    def test_unknown_term_scores_zero(self):
        bm25 = BM25().fit(self.DOCS)
        self.assertEqual(sum(bm25.scores("zzzzz")), 0.0)

    def test_empty_query(self):
        bm25 = BM25().fit(self.DOCS)
        self.assertEqual(bm25.scores("，。！"), [0.0] * 3)

    def test_empty_corpus(self):
        self.assertEqual(BM25().fit([]).scores("x"), [])

    def test_idf_is_non_negative(self):
        bm25 = BM25().fit(self.DOCS)
        self.assertTrue(all(v >= 0 for v in bm25.idf.values()))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            BM25(k1=0)
        with self.assertRaises(ValueError):
            BM25(b=1.5)


class TestRrf(unittest.TestCase):
    def test_combines_two_paths(self):
        fused = rrf_fuse([[0, 1, 2], [2, 0, 1]])
        self.assertIn(0, fused)
        self.assertIn(2, fused)

    def test_document_in_both_paths_wins(self):
        fused = rrf_fuse([[0, 1], [0, 1]])
        self.assertGreater(fused[0], fused[1])

    def test_weights(self):
        fused = rrf_fuse([[0], [1]], weights=[5.0, 1.0])
        self.assertGreater(fused[0], fused[1])

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            rrf_fuse([[0]], k=0)

    def test_empty(self):
        self.assertEqual(rrf_fuse([[], []]), {})


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from minirag.corpus import DOCUMENTS
        cls.retriever = Retriever(
            chunks=[c for d in DOCUMENTS
                    for c in chunk_document(d.id, d.title, d.text, size=200)],
            embedder=HashingEmbedder(dim=512),
        )

    def test_modes_return_hits(self):
        for mode in ("vector", "bm25", "hybrid"):
            with self.subTest(mode=mode):
                hits = self.retriever.search("BM25 稀疏检索", mode=mode, top_k=3)
                self.assertEqual(len(hits), 3)
                self.assertTrue(all(h.mode == mode for h in hits))
                self.assertEqual([h.rank for h in hits], [1, 2, 3])

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.retriever.search("x", mode="magic")

    def test_bm25_finds_exact_term(self):
        hits = self.retriever.search("HNSW 分层可导航小世界图", mode="bm25", top_k=1)
        self.assertEqual(hits[0].doc_id, "note-hnsw")

    def test_doc_ranking_dedupes(self):
        ranking = self.retriever.doc_ranking("检索", mode="hybrid", top_k=5)
        self.assertEqual(len(ranking), len(set(ranking)), "文档级排名不能有重复")

    def test_doc_ranking_caps_per_document(self):
        ranking = self.retriever.doc_ranking("检索 记忆 评测 工具", mode="hybrid",
                                             top_k=6, max_per_doc=1)
        self.assertLessEqual(len(ranking), 6)
        self.assertEqual(len(ranking), len(set(ranking)))

    def test_empty_retriever(self):
        empty = Retriever(chunks=[], embedder=HashingEmbedder(dim=32))
        self.assertEqual(empty.search("x"), [])
        self.assertEqual(empty.doc_ranking("x"), [])

    def test_hit_serialisable(self):
        import json
        hit = self.retriever.search("BM25", mode="bm25", top_k=1)[0]
        self.assertEqual(json.loads(json.dumps(hit.to_dict()))["mode"], "bm25")


class TestVectorIndex(unittest.TestCase):
    def test_scores_length(self):
        index = VectorIndex(HashingEmbedder(dim=64)).fit_texts(["a 检索", "b 记忆"])
        self.assertEqual(len(index.scores("检索")), 2)

    def test_empty(self):
        index = VectorIndex(HashingEmbedder(dim=64)).fit_texts([])
        self.assertEqual(index.scores("x"), [])


if __name__ == "__main__":
    unittest.main()
