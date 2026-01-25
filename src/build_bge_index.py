"""
build_bge_index.py

构建 PubMed RCT 文本块的向量索引（BGE + Chroma）。

主要功能：
1. 加载文本块数据（train / validation / test）
2. 加载 BGE 嵌入模型
3. 批量生成文本块向量
4. 使用 Chroma 构建持久化向量索引
5. 输出索引统计信息（JSON）

对应任务：
- 1 嵌入模型选择与加载
- 2 向量数据库配置与索引构建
"""

from pathlib import Path
from datetime import datetime
import argparse
import json

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb


# =========================
# 1. 数据加载与模型加载
# =========================

def load_chunks(splits: list[str]) -> pd.DataFrame:
    """
    从 data/ 目录加载指定 split 的文本块 CSV，并合并为一个 DataFrame。

    参数：
        splits: 需要加载的数据划分列表，例如 ["train"] 或 ["train", "validation"]。

    返回：
        包含所有文本块的 DataFrame，每行代表一个文本块，并带有 split 字段。
    """
    data_dir = Path("data")
    dfs: list[pd.DataFrame] = []

    for split in splits:
        csv_path = data_dir / f"pubmed_rct_{split}_chunks.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到文本块文件：{csv_path}")

        df = pd.read_csv(csv_path)
        # 记录该行所属的数据划分，便于后续统计与过滤
        df["split"] = split
        dfs.append(df)

    if not dfs:
        raise ValueError("未提供有效的 split 或对应数据文件不存在。")

    return pd.concat(dfs, ignore_index=True)


def load_bge_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    """
    加载 BGE 嵌入模型（SentenceTransformer 封装）。

    该模型用于：
    - 对文本块生成向量（passage embedding）
    - 对查询生成向量（query embedding，后续检索阶段使用）

    参数：
        model_name: Hugging Face 上的模型名称。

    返回：
        SentenceTransformer 模型实例。
    """
    print(f"[embedder] 加载嵌入模型：{model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"[embedder] 模型加载完成，向量维度：{dim}")
    return model


# =========================
# 2. 构建 Chroma 向量索引
# =========================

def build_chroma_index(
    chunks_df: pd.DataFrame,
    model: SentenceTransformer,
    collection_name: str = "pubmed_rct_bge",
    persist_dir: str = "./chroma_medrag_bge",
    batch_size: int = 64,
):
    """
    使用 Chroma 构建持久化向量索引。

    步骤：
    1. 初始化 Chroma 持久化客户端
    2. 创建 / 获取集合（使用余弦相似度）
    3. 按批次对文本块生成嵌入向量，并写入集合

    参数：
        chunks_df: 文本块数据，每行包含至少 text、doc_id、chunk_index 等字段
        model: 已加载的 BGE 嵌入模型
        collection_name: Chroma 集合名称
        persist_dir: Chroma 持久化目录
        batch_size: 每批嵌入计算的文本数量，内存不足时可调小

    返回：
        Chroma 集合对象（用于后续查询或统计）
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))

    # 使用 cosine 作为相似度度量空间
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if "text" not in chunks_df.columns:
        raise ValueError("文本块数据中缺少 'text' 列，请检查 CSV。")

    total = len(chunks_df)
    print(f"[index] 需要写入的文本块总数：{total}")

    for start in tqdm(range(0, total, batch_size), desc="[index] embedding & add"):
        end = min(start + batch_size, total)
        batch = chunks_df.iloc[start:end]

        # 文本内容
        texts = batch["text"].astype(str).tolist()

        # 按 BGE 官方推荐，为检索任务添加前缀指令
        passages = [
            f"Represent this passage for retrieval: {t}" for t in texts
        ]

        # 生成归一化后的向量，适合 cosine 相似度
        embeddings = model.encode(
            passages,
            batch_size=len(passages),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        ids = []
        metadatas = []

        for _, row in batch.iterrows():
            # 优先使用 doc_id；如不存在则回退到 chunk_id
            doc_id = str(row.get("doc_id", row.get("chunk_id", "")))
            chunk_index = int(row.get("chunk_index", 0))

            # 唯一 ID：doc_id-chunk_index
            chunk_id = f"{doc_id}-{chunk_index}"
            ids.append(chunk_id)

            meta = {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "total_chunks": int(row.get("total_chunks", 1)),
                "split": row.get("split", "train"),
            }

            # 可选元数据字段
            if "token_count" in row:
                try:
                    meta["token_count"] = int(row["token_count"])
                except Exception:
                    pass

            if "source_title" in row and isinstance(row["source_title"], str):
                if row["source_title"].strip():
                    meta["source_title"] = row["source_title"]

            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(
        f"[index] 索引构建完成，集合 '{collection_name}' "
        f"当前向量数：{collection.count()}"
    )
    return collection


# =========================
# 3. 索引统计信息输出
# =========================

def save_index_stats(
    chunks_df: pd.DataFrame,
    model: SentenceTransformer,
    collection_name: str,
    output_path: Path,
) -> None:
    """
    保存索引相关统计信息到 JSON 文件。

    对应任务中的 stats 字段设计：
        - collection_name
        - total_chunks
        - embedding_model
        - embedding_dimension
        - index_built_at
        - chunk_size_stats
        - metadata_fields
    """
    stats = {
        "collection_name": collection_name,
        "total_chunks": int(len(chunks_df)),
        "embedding_model": model.name_or_path
        if hasattr(model, "name_or_path")
        else "BAAI/bge-small-en-v1.5",
        "embedding_dimension": int(
            model.get_sentence_embedding_dimension()
        ),
        "index_built_at": datetime.now().isoformat(),
        "chunk_size_stats": {},
        "metadata_fields": [],
    }

    if "token_count" in chunks_df.columns:
        stats["chunk_size_stats"] = {
            "mean": float(chunks_df["token_count"].mean()),
            "max": int(chunks_df["token_count"].max()),
            "min": int(chunks_df["token_count"].min()),
        }

    # 元数据字段列表用于文档说明，便于后续过滤查询设计
    metadata_fields = ["doc_id", "chunk_index", "total_chunks", "split"]
    if "token_count" in chunks_df.columns:
        metadata_fields.append("token_count")
    if "source_title" in chunks_df.columns:
        metadata_fields.append("source_title")

    stats["metadata_fields"] = metadata_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[stats] 索引统计信息已保存到：{output_path}")


# =========================
# 4. 命令行入口
# =========================

def main() -> None:
    """
    命令行入口。

    当前行为：
    - 解析参数（数据划分 / 集合名称 / 持久化目录 / batch_size）
    - 加载文本块数据
    - 加载 BGE 嵌入模型
    - 构建 Chroma 向量索引
    - 输出索引统计信息
    """
    parser = argparse.ArgumentParser(
        description="构建 PubMed RCT 文本块的 BGE + Chroma 向量索引。"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="使用的数据划分，逗号分隔，例如 'train' 或 'train,validation'。",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="pubmed_rct_bge",
        help="Chroma 集合名称。",
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./chroma_medrag_bge",
        help="Chroma 持久化目录。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="每批嵌入计算的文本数量，内存不足时可适当调小。",
    )

    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"[config] 使用的数据划分：{splits}")

    # 1) 加载文本块数据
    chunks_df = load_chunks(splits)
    print(f"[data] 文本块数量：{len(chunks_df)}")
    print(f"[data] 列名：{list(chunks_df.columns)}")

    # 2) 加载嵌入模型
    model = load_bge_model("BAAI/bge-small-en-v1.5")

    # 3) 构建 Chroma 向量索引
    collection = build_chroma_index(
        chunks_df=chunks_df,
        model=model,
        collection_name=args.collection_name,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
    )

    # 4) 输出索引统计信息
    stats_path = Path("data") / "pubmed_rct_bge_index_stats.json"
    save_index_stats(
        chunks_df=chunks_df,
        model=model,
        collection_name=args.collection_name,
        output_path=stats_path,
    )

    print(
        f"[done] 索引构建与统计信息输出完成。"
        f" Chroma 路径：{args.persist_dir}，"
        f"集合 '{args.collection_name}' 向量数：{collection.count()}"
    )


if __name__ == "__main__":
    main()
