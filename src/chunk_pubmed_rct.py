"""
chunk_pubmed_rct.py

功能：
- 从 PubMed RCT 数据集 (armanc/pubmed-rct20k) 构造“摘要级文档 DataFrame”
- 再根据策略，将每篇文献切分为适合向量化和检索的文本块（chunk）
- 当前版本：采用“整体不分割”策略（一篇摘要 = 一个 chunk）

输出：
- data/pubmed_rct_train_chunks.csv  文本块数据集
- data/pubmed_rct_train_chunks_stats.json  处理配置和统计信息
"""

from datasets import load_dataset
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import argparse

import pandas as pd


# ===== 1. 配置与分块工具 =====

@dataclass
class ChunkConfig:
    """
    文本分块的配置

    chunk_size / chunk_overlap 目前主要给未来“滑动窗口分割”用，
    当前 split_mode="no_split" 时，不会真正用到。
    """
    chunk_size: int = 512
    chunk_overlap: int = 64
    split_mode: str = "no_split"  # 可选: "no_split" 或 "sliding"


class PubMedChunker:
    """
    负责把“一篇文献（full_text）”切成多个 chunk。
    目前支持两种模式：
    - no_split: 整篇不分割，一篇文献 = 一个 chunk
    - sliding: 采用简单的“按单词数”的滑动窗口切分（备用）
    """

    def __init__(self, config: ChunkConfig):
        self.config = config

    def _count_tokens(self, text: str) -> int:
        """
        统计 token 数的函数。

        这里先使用非常简单的近似：
        - 按空格切分单词，单词数 ≈ token 数
        未来如果想对齐某个具体模型的 tokenizer，
        可以在这里替换为真正的 tokenizer 调用。
        """
        if not text:
            return 0
        return len(text.strip().split())

    def _split_text_sliding(self, text: str) -> List[str]:
        """
        简单的“按单词滑动窗口”切分：
        - 先把文本按空格拆成 words
        - 按 chunk_size 个单词一段
        - 相邻 chunk 之间保留 chunk_overlap 个单词重叠
        """
        words = text.strip().split()
        if not words:
            return []

        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # 文本本身就不长，直接作为一个 chunk
        if len(words) <= size:
            return [" ".join(words)]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end >= len(words):
                break

            # 下一个窗口往前“回退” overlap 个单词，制造重叠
            start = end - overlap

        return chunks

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        把一篇文献切成一个或多个 chunk。

        输入 document 结构：
        {
            "doc_id": str,     # 文献 ID（这里用 abstract_id）
            "title": str,      # 标题（本数据集没有标题，暂时为空字符串）
            "full_text": str,  # 摘要全文（多个句子拼接）
        }

        输出为 chunk 列表，每个元素是一个 dict，包含：
        - chunk_id: 本块的唯一 ID
        - text: 本块的文本内容
        - doc_id: 原文 ID
        - chunk_index: 本块在原文中的序号（从 0 开始）
        - total_chunks: 原文被分成的总块数
        - source_title: 原文标题
        - token_count: 本块的 token 近似数
        """
        full_text = document["full_text"]
        title = document.get("title", "")
        doc_id = str(document["doc_id"])

        # 1) 根据策略决定是否切分
        if self.config.split_mode == "no_split":
            texts = [full_text]
        else:
            texts = self._split_text_sliding(full_text)

        # 2) 生成 chunk 元数据
        chunks: List[Dict] = []
        total_chunks = len(texts)

        for i, text in enumerate(texts):
            # 如果只有一个 chunk，直接用 doc_id 当 chunk_id；
            # 如果有多个 chunk，则在 doc_id 后面加序号。
            if total_chunks == 1:
                chunk_id = doc_id
            else:
                chunk_id = f"{doc_id}-{i}"

            chunk_token_count = self._count_tokens(text)

            chunk_data = {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "source_title": title,
                "token_count": chunk_token_count,
            }
            chunks.append(chunk_data)

        return chunks


# ===== 2. 从 PubMed RCT 构造“摘要级文档” =====

def build_docs_from_pubmed_rct(split: str = "train") -> pd.DataFrame:
    """
    从 armanc/pubmed-rct20k 的某个 split 构造“摘要级文档 DataFrame”：

    思路：
    - 原始数据是“句子级”：每行是一个句子，有 abstract_id 和 sentence_id
    - 我们按 abstract_id 分组，再按 sentence_id 排序，拼成 full_text
    - 本数据集没有 title 字段，所以 title 暂时留空 ""
    """
    print(f"[build_docs] 加载数据集 armanc/pubmed-rct20k, split = {split} ...")
    ds = load_dataset("armanc/pubmed-rct20k")
    data = ds[split]

    # 转成 DataFrame 方便 groupby 操作
    df = data.to_pandas()

    # 按 abstract_id 分组，并按 sentence_id 排序
    grouped = df.sort_values(["abstract_id", "sentence_id"]).groupby("abstract_id")

    docs = []
    for abstract_id, g in grouped:
        sentences = [str(t) for t in g["text"].tolist()]
        full_text = " ".join(sentences)

        docs.append(
            {
                "doc_id": str(abstract_id),  # 这里直接用 abstract_id 作为文献 ID
                "title": "",                 # 数据集中没有标题，用空字符串占位
                "full_text": full_text,
            }
        )

    df_docs = pd.DataFrame(docs)
    return df_docs


# ===== 3. 主流程：分块 + 保存 + 统计 =====

def main():
    # 使用 argparse 从命令行读取要处理的数据 split
    parser = argparse.ArgumentParser(description="Chunk PubMed RCT dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="要处理的数据子集（train / validation / test），默认 train",
    )
    args = parser.parse_args()
    data_split = args.split

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 构造摘要级文档表
    df_docs = build_docs_from_pubmed_rct(split=data_split)
    print(f"[main] 共构造摘要级文献数（split={data_split}）：{len(df_docs)}")

    # 2) 初始化分块配置与分块器
    config = ChunkConfig(
        chunk_size=512,
        chunk_overlap=64,
        split_mode="no_split",  # 当前任务采用“整体不分割”策略
    )
    chunker = PubMedChunker(config)

    # 3) 遍历每篇文献，生成 chunk
    all_chunks: List[Dict] = []

    for _, row in df_docs.iterrows():
        document = {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "full_text": row["full_text"],
        }
        chunks = chunker.chunk_document(document)
        all_chunks.extend(chunks)

    chunks_df = pd.DataFrame(all_chunks)
    print(f"[main] 生成的文本块总数：{len(chunks_df)}")

    # 4) 计算统计信息
    stats = {
        "processed_date": pd.Timestamp.now().isoformat(),
        "data_split": data_split,
        "original_documents": len(df_docs),
        "total_chunks": len(chunks_df),
        "chunks_per_doc": len(chunks_df) / len(df_docs) if len(df_docs) > 0 else 0,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "split_mode": config.split_mode,
    }

    if not chunks_df.empty:
        stats.update(
            {
                "min_token_count": int(chunks_df["token_count"].min()),
                "max_token_count": int(chunks_df["token_count"].max()),
                "avg_token_count": float(chunks_df["token_count"].mean()),
            }
        )

    # 5) 保存结果文件（根据 split 命名）
    chunks_path = output_dir / f"pubmed_rct_{data_split}_chunks.csv"
    stats_path = output_dir / f"pubmed_rct_{data_split}_chunks_stats.json"

    chunks_df.to_csv(chunks_path, index=False)
    pd.Series(stats).to_json(stats_path, indent=2, force_ascii=False)

    print(f"[main] 文本块数据已保存到：{chunks_path}")
    print(f"[main] 统计信息已保存到：{stats_path}")

    # 6) 预览前几行，方便人工检查（对应任务 4：预览结果）
    print("\n[preview] 前 5 条文本块：")
    print(chunks_df.head(5))

    # 7) 简单质量验证：多块文献数量（对应任务 5：基础质量检查）
    if "total_chunks" in chunks_df.columns:
        num_multi_chunk_docs = (chunks_df["total_chunks"] > 1).sum()
        print(f"\n[quality] total_chunks > 1 的文献数量：{num_multi_chunk_docs}")
    print("[quality] 质量检查（基础版）完成。")

if __name__ == "__main__":
    main()	
