"""
keyword_bm25.py

基于文本块数据构建 BM25 关键词索引，并提供简单的检索接口。
使用之前已生成的 PubMed RCT 文本块 CSV 文件：
- data/pubmed_rct_train_chunks.csv
- data/pubmed_rct_validation_chunks.csv
- data/pubmed_rct_test_chunks.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import re

import pandas as pd
from rank_bm25 import BM25Okapi


# 默认同时加载 train / validation / test 三个分割的文本块
DEFAULT_CHUNK_FILES = [
    "data/pubmed_rct_train_chunks.csv",
    "data/pubmed_rct_validation_chunks.csv",
    "data/pubmed_rct_test_chunks.csv",
]


def simple_tokenize(text: str) -> List[str]:
    """
    简单英文分词函数：
    - 全部转小写
    - 用正则提取英文字母和数字
    """
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


@dataclass
class BM25Document:
    """BM25 使用的文档信息，方便后续和向量检索结果对齐。"""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]


class BM25KeywordIndex:
    """
    BM25 关键词索引：
    - 负责加载文本块数据
    - 构建 BM25Okapi 模型
    - 提供 search() 方法返回 Top-K 文档
    """

    def __init__(self, chunk_files: Optional[List[str]] = None):
        # 如果没有传入，就用默认的三份 CSV
        self.chunk_files = chunk_files or DEFAULT_CHUNK_FILES
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[BM25Document] = []
        self._corpus_tokens: List[List[str]] = []

    # ---------- 索引构建 ----------

    def build(self) -> None:
        """从多个 CSV 读取文本块数据，构建 BM25 索引。"""
        print("[bm25] 加载文本块数据：")
        dfs: List[pd.DataFrame] = []

        for path in self.chunk_files:
            print(f"  - {path}")
            df_part = pd.read_csv(path)
            dfs.append(df_part)

        if not dfs:
            raise ValueError("未找到任何文本块文件，请检查文件路径配置。")

        df = pd.concat(dfs, ignore_index=True)

        # 这里假设有 text / chunk_id / doc_id 字段，其他字段作为元数据保留
        required_cols = {"text", "chunk_id", "doc_id"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"BM25 需要的字段缺失，至少应包含：{required_cols}，当前字段：{list(df.columns)}"
            )

        self._documents = []
        self._corpus_tokens = []

        for _, row in df.iterrows():
            text = str(row["text"])
            tokens = simple_tokenize(text)
            self._corpus_tokens.append(tokens)

            # 把除了 text 以外的列都丢进 metadata，方便之后做过滤 / 打印
            metadata = row.to_dict()
            metadata.pop("text", None)

            doc = BM25Document(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                text=text,
                metadata=metadata,
            )
            self._documents.append(doc)

        print(f"[bm25] 语料文档数：{len(self._documents)}")

        # 构建 BM25 模型
        self._bm25 = BM25Okapi(self._corpus_tokens)
        print("[bm25] BM25 索引构建完成。")

    # ---------- 检索 ----------

    def is_ready(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = 10) -> List[BM25Document]:
        """
        使用 BM25 进行关键词检索。

        Args:
            query: 原始查询字符串
            top_k: 返回前 top_k 条结果

        Returns:
            BM25Document 列表，按得分从高到低排序
        """
        if not self._bm25:
            raise RuntimeError("BM25 索引尚未构建，请先调用 build()。")

        tokens = simple_tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # 取得分最高的前 top_k 个下标
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[BM25Document] = []
        for i in top_indices:
            results.append(self._documents[i])

        return results


# ---------- 简单自测 ----------

def main():
    index = BM25KeywordIndex()
    index.build()

    while True:
        q = input("\n[bm25-demo] 请输入英文查询（回车退出）：").strip()
        if not q:
            print("[bm25-demo] 退出。")
            break

        docs = index.search(q, top_k=5)
        print(f"[bm25-demo] 命中数量：{len(docs)}")
        for rank, d in enumerate(docs, start=1):
            print(f"\nTop {rank}")
            print(f"  chunk_id: {d.chunk_id}")
            print(f"  doc_id  : {d.doc_id}")
            print(f"  text    : {d.text[:200]}...")


if __name__ == "__main__":
    main()