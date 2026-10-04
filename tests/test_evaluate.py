"""评测与生成层测试 —— 重点是"指标算得对不对"和"引用能不能被抓假"。"""

from __future__ import annotations

import json
import unittest

from minirag.corpus import DOCUMENTS, QUERIES
from minirag.embed import HashingEmbedder
from minirag.evaluate import (
    ablation_retrieval,
    build_chunks,
    build_retriever,
    evaluate,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    render_tables,
)
from minirag.rag import ExtractiveMockLLM, RagPipeline, citation_accuracy


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a"], 1), 1.0)
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a"], 0), 0.0)
        self.assertEqual(recall_at_k(["a", "b"], ["c"], 5), 0.0)
        self.assertAlmostEqual(recall_at_k(["a", "b"], ["a", "b"], 5), 1.0)

    def test_recall_partial(self):
        self.assertAlmostEqual(recall_at_k(["a", "x"], ["a", "b"], 5), 0.5)

    def test_recall_empty_relevant(self):
        self.assertEqual(recall_at_k(["a"], [], 5), 0.0)

    def test_hit(self):
        self.assertEqual(hit_at_k(["a", "b"], ["b"], 2), 1.0)
        self.assertEqual(hit_at_k(["a", "b"], ["b"], 1), 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["a", "b"], ["a"]), 1.0)
        self.assertEqual(reciprocal_rank(["x", "a"], ["a"]), 0.5)
        self.assertEqual(reciprocal_rank(["x", "y"], ["a"]), 0.0)
        self.assertAlmostEqual(reciprocal_rank(["x", "a", "b"], ["b"]), 1 / 3)


class TestBuilders(unittest.TestCase):
    def test_build_chunks_counts(self):
        chunks = build_chunks(DOCUMENTS, strategy="recursive", size=200, overlap=40)
        self.assertGreater(len(chunks), len(DOCUMENTS), "长文档必须被切成多块")

    def test_strategies_give_different_chunk_counts(self):
        counts = {s: len(build_chunks(DOCUMENTS[:5], strategy=s, size=200))
                  for s in ("fixed", "recursive", "heading")}
        self.assertGreater(len(set(counts.values())), 1, "三种策略应该产生不同的切分")

    def test_build_retriever(self):
        retriever = build_retriever(DOCUMENTS[:3], embedder=HashingEmbedder(dim=64))
        self.assertTrue(retriever.chunks)


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = build_retriever(DOCUMENTS, strategy="recursive")
        cls.report = evaluate(cls.retriever, QUERIES, mode="bm25", k=5, label="test")

    def test_report_shape(self):
        self.assertEqual(self.report.total, len(QUERIES))
        self.assertEqual(self.report.k, 5)
        self.assertTrue(0.0 <= self.report.recall <= 1.0)
        self.assertTrue(0.0 <= self.report.mrr <= 1.0)

    def test_by_kind_covers_all_kinds(self):
        kinds = set(self.report.by_kind())
        self.assertEqual(kinds, {"keyword", "semantic", "mixed"})

    def test_by_kind_counts_sum(self):
        total = sum(v["n"] for v in self.report.by_kind().values())
        self.assertEqual(total, len(QUERIES))

    def test_failures_are_real_misses(self):
        for item in self.report.failures():
            self.assertFalse(set(item.retrieved[:5]) & set(item.relevant))

    def test_to_dict_serialisable(self):
        payload = json.loads(json.dumps(self.report.to_dict()))
        self.assertEqual(payload["mode"], "bm25")

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            evaluate(self.retriever, QUERIES, mode="magic")


class TestAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = ablation_retrieval(DOCUMENTS, QUERIES, k=5)

    def test_all_combinations_present(self):
        for strategy in ("fixed", "recursive", "heading"):
            for mode in ("vector", "bm25", "hybrid"):
                self.assertIn(f"{strategy}/{mode}", self.reports)

    def test_recall_in_range(self):
        for key, report in self.reports.items():
            with self.subTest(key=key):
                self.assertTrue(0.0 <= report.recall <= 1.0)

    def test_lexical_beats_toy_vector(self):
        """本仓库的哈希嵌入无语义能力，所以纯词面检索应该更强。

        这不是"混合检索没用"，而是"两路检索必须有**互补**的信号才有融合价值"。
        换成真实嵌入模型（--live）时这个断言不成立。
        """
        bm25 = self.reports["recursive/bm25"]
        vector = self.reports["recursive/vector"]
        self.assertGreaterEqual(bm25.recall, vector.recall)

    def test_render_tables_contains_all_sections(self):
        text = render_tables(self.reports, k=5)
        self.assertIn("表 1", text)
        self.assertIn("表 2", text)
        self.assertIn("表 3", text)
        for mode in ("bm25", "vector", "hybrid"):
            self.assertIn(f"`{mode}`", text)

    def test_ablation_is_deterministic(self):
        again = ablation_retrieval(DOCUMENTS, QUERIES, k=5)
        for key in self.reports:
            self.assertEqual(self.reports[key].recall, again[key].recall, key)


class TestRagPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = build_retriever(DOCUMENTS, strategy="recursive")
        cls.pipeline = RagPipeline(retriever=cls.retriever, llm=ExtractiveMockLLM(), top_k=4)

    def test_answer_has_citations(self):
        answer = self.pipeline.answer("BM25 稀疏检索擅长什么")
        self.assertTrue(answer.citations)
        self.assertTrue(answer.cited_indexes)
        self.assertEqual(answer.invalid_indexes, [])
        self.assertTrue(answer.grounded)

    def test_context_budget_respected(self):
        pipeline = RagPipeline(retriever=self.retriever, llm=ExtractiveMockLLM(),
                               top_k=10, max_context_chars=300)
        answer = pipeline.answer("检索")
        self.assertLessEqual(answer.context_chars, 300 + 200)

    def test_citation_indexes_are_sequential(self):
        answer = self.pipeline.answer("记忆")
        self.assertEqual([c.index for c in answer.citations],
                         list(range(1, len(answer.citations) + 1)))

    def test_citation_accuracy_of_good_answer(self):
        answer = self.pipeline.answer("检索")
        self.assertEqual(answer.citation_accuracy, 1.0)

    def test_fabricated_citation_is_detected(self):
        class LyingLLM(ExtractiveMockLLM):
            def generate(self, *, system: str, user: str) -> str:
                return "根据资料 [1]，结论如此 [99]。"

        pipeline = RagPipeline(retriever=self.retriever, llm=LyingLLM(), top_k=2)
        answer = pipeline.answer("检索")
        self.assertIn(99, answer.invalid_indexes)
        self.assertFalse(answer.grounded)
        self.assertLess(answer.citation_accuracy, 1.0)

    def test_no_citation_scores_zero(self):
        class SilentLLM(ExtractiveMockLLM):
            def generate(self, *, system: str, user: str) -> str:
                return "我觉得是这样。"

        pipeline = RagPipeline(retriever=self.retriever, llm=SilentLLM(), top_k=2)
        answer = pipeline.answer("检索")
        self.assertEqual(answer.citation_accuracy, 0.0)

    def test_empty_retriever_returns_safe_answer(self):
        empty = build_retriever([], strategy="recursive")
        pipeline = RagPipeline(retriever=empty, llm=ExtractiveMockLLM())
        answer = pipeline.answer("随便问")
        self.assertIn("无法回答", answer.answer)

    def test_citation_accuracy_aggregate(self):
        answers = [self.pipeline.answer(q.text) for q in QUERIES[:5]]
        self.assertGreaterEqual(citation_accuracy(answers), 0.0)

    def test_citation_accuracy_empty(self):
        self.assertEqual(citation_accuracy([]), 0.0)

    def test_answer_to_dict(self):
        payload = self.pipeline.answer("检索").to_dict()
        self.assertIn("citation_accuracy", json.loads(json.dumps(payload)))


if __name__ == "__main__":
    unittest.main()
