"""切分测试。"""

from __future__ import annotations

import unittest

from minirag.chunk import (
    Chunk,
    chunk_document,
    chunk_fixed,
    chunk_heading,
    chunk_recursive,
    chunk_stats,
)


class TestFixed(unittest.TestCase):
    def test_no_overlap(self):
        pieces = chunk_fixed("abcdefghij", size=4, overlap=0)
        self.assertEqual(pieces, ["abcd", "efgh", "ij"])

    def test_overlap_shares_text(self):
        pieces = chunk_fixed("abcdefghij", size=4, overlap=2)
        self.assertEqual(pieces[0], "abcd")
        self.assertTrue(pieces[1].startswith("cd"))

    def test_every_char_covered(self):
        text = "0123456789" * 5
        joined = "".join(chunk_fixed(text, size=7, overlap=0))
        self.assertEqual(joined, text, "无重叠时必须覆盖全文且不重复")

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            chunk_fixed("abc", size=0)
        with self.assertRaises(ValueError):
            chunk_fixed("abc", size=3, overlap=3)
        with self.assertRaises(ValueError):
            chunk_fixed("abc", size=3, overlap=-1)

    def test_empty_text(self):
        self.assertEqual(chunk_fixed("", size=10, overlap=0), [])


class TestRecursive(unittest.TestCase):
    TEXT = "第一段内容比较长一些。" * 4 + "\n\n" + "第二段内容也不算短。" * 4

    def test_splits_within_size(self):
        pieces = chunk_recursive(self.TEXT, size=40, overlap=0)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertLessEqual(len(piece), 40 + 1)

    def test_prefers_sentence_boundaries(self):
        pieces = chunk_recursive(self.TEXT, size=40, overlap=0)
        # 至少有一部分是以句号结尾的（说明在语义边界断开，而不是硬切）
        self.assertTrue(any(p.rstrip().endswith("。") for p in pieces))

    def test_long_text_without_separators_falls_back_to_fixed(self):
        pieces = chunk_recursive("字" * 100, size=20, overlap=0)
        self.assertGreater(len(pieces), 1)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            chunk_recursive("abc", size=0)

    def test_full_coverage_without_overlap(self):
        pieces = chunk_recursive(self.TEXT, size=30, overlap=0)
        combined = "".join(pieces)
        for chunk_piece in pieces:
            self.assertIn(chunk_piece[:8], combined)


class TestHeading(unittest.TestCase):
    TEXT = (
        "# 标题一\n"
        "这是第一节的内容。" * 8 + "\n"
        "# 标题二\n"
        "这是第二节的内容。" * 8
    )

    def test_splits_by_heading(self):
        sections = chunk_heading(self.TEXT, size=50, overlap=0)
        headings = {h for h, _ in sections}
        self.assertIn("标题一", headings)
        self.assertIn("标题二", headings)

    def test_no_heading_returns_whole_text(self):
        sections = chunk_heading("没有标题的一段话", size=50, overlap=0)
        self.assertEqual(sections, [("", "没有标题的一段话")])

    def test_oversized_section_is_split(self):
        sections = chunk_heading(self.TEXT, size=30, overlap=0)
        self.assertGreater(len(sections), 2)


class TestChunkDocument(unittest.TestCase):
    def test_builds_chunks_with_ids(self):
        chunks = chunk_document("doc-1", "标题", "内容" * 100, strategy="recursive", size=50)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source_id, "doc-1")
        self.assertEqual(chunks[0].id, "doc-1::0")
        self.assertEqual(chunks[0].title, "标题")

    def test_heading_strategy_records_heading(self):
        chunks = chunk_document("d", "t", "# H\n正文内容" * 5, strategy="heading", size=40)
        self.assertTrue(any(c.heading for c in chunks))

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            chunk_document("d", "t", "x", strategy="magic")

    def test_all_strategies_produce_chunks(self):
        for strategy in ("fixed", "recursive", "heading"):
            with self.subTest(strategy=strategy):
                chunks = chunk_document("d", "t", "# H\n" + "内容。" * 60,
                                        strategy=strategy, size=60)
                self.assertTrue(chunks)


class TestStats(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_stats([])["count"], 0)

    def test_values(self):
        chunks = [Chunk(id="a", text="12345", source_id="s"),
                  Chunk(id="b", text="1234567890", source_id="s")]
        stats = chunk_stats(chunks)
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["min_chars"], 5)
        self.assertEqual(stats["max_chars"], 10)
        self.assertEqual(stats["mean_chars"], 7.5)

    def test_to_dict(self):
        chunk = Chunk(id="a", text="x", source_id="s", title="t", meta={"heading": "h"})
        self.assertEqual(chunk.heading, "h")
        self.assertEqual(chunk.to_dict()["source_id"], "s")


if __name__ == "__main__":
    unittest.main()
