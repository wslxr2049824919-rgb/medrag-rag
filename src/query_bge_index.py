"""
query_bge_index.py

用于对已构建的 PubMed RCT BGE + Chroma 向量索引做简单质量验证：
- 手动查询：输入英文医学问题，查看检索结果
- 自检：随机抽取索引中的文本块作为查询，检查自相似性
"""

from pathlib import Path
import random
import argparse
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import chromadb


def load_bge_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    """
    加载查询阶段使用的 BGE 嵌入模型。
    必须与建索引阶段的模型保持一致。
    """
    print(f"[embedder] 加载嵌入模型：{model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"[embedder] 模型加载完成，向量维度：{dim}")
    return model


def get_chroma_collection(
    persist_dir: str = "./chroma_medrag_bge",
    collection_name: str = "pubmed_rct_bge",
):
    """
    连接已构建好的 Chroma 持久化集合。
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"未找到 Chroma 持久化目录：{persist_path}，"
            f"请确认已执行 build_bge_index.py。"
        )

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_collection(name=collection_name)
    count = collection.count()
    print(f"[chroma] 已连接集合 '{collection_name}'，当前向量数：{count}")
    return collection


def embed_query(
    query_text: str,
    model: SentenceTransformer,
) -> List[List[float]]:
    """
    将查询文本编码为向量，返回二维 list。
    按 BGE 推荐，增加 query 前缀。
    """
    text = query_text.strip()
    if not text:
        raise ValueError("查询文本为空。")

    prefixed = f"Represent this query for retrieval: {text}"
    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.tolist()


def run_query(
    query_text: str,
    model: SentenceTransformer,
    collection,
    n_results: int = 5,
    where_filter: Dict[str, Any] | None = None,
) -> None:
    """
    在 Chroma 集合上执行一次查询，并打印前 n_results 条结果。
    """
    text = query_text.strip()
    if not text:
        print("[query] 查询文本为空，已跳过。")
        return

    print(f"\n[query] 查询文本：{text}")
    query_embeddings = embed_query(text, model)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=where_filter,
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    print(f"[query] 返回结果数量：{len(ids)}\n")

    for rank, (cid, dist, doc, meta) in enumerate(
        zip(ids, distances, documents, metadatas), start=1
    ):
        doc_id = meta.get("doc_id", "N/A")
        chunk_index = meta.get("chunk_index", "N/A")
        split = meta.get("split", "N/A")
        token_count = meta.get("token_count", "N/A")

        snippet = doc[:300] + ("..." if len(doc) > 300 else "")

        print(f"Top {rank}")
        print(f"  chunk_id    : {cid}")
        print(f"  doc_id      : {doc_id}")
        print(f"  chunk_index : {chunk_index}")
        print(f"  split       : {split}")
        print(f"  token_count : {token_count}")
        print(f"  distance    : {dist:.4f}")
        print(f"  text        : {snippet}")
        print("")


def self_check(
    model: SentenceTransformer,
    collection,
    n_results: int = 5,
    sample_limit: int = 200,
) -> None:
    """
    自检：从索引中随机抽取一个文本块作为查询，检查相似检索是否正常。
    """
    print(f"\n[self-check] 从索引中预取最多 {sample_limit} 条记录。")

    fetched = collection.get(
        limit=sample_limit,
        include=[ "documents", "metadatas"],
    )

    docs = fetched.get("documents", [])
    metadatas = fetched.get("metadatas", [])
    ids = fetched.get("ids", [])

    if not docs:
        print("[self-check] 未获取到任何文档。")
        return

    idx = random.randrange(len(docs))
    query_text = docs[idx]
    query_meta = metadatas[idx]
    query_id = ids[idx]

    print("\n[self-check] 选中的原始文本块：")
    print(f"  chunk_id    : {query_id}")
    print(f"  doc_id      : {query_meta.get('doc_id', 'N/A')}")
    print(f"  chunk_index : {query_meta.get('chunk_index', 'N/A')}")
    print(f"  split       : {query_meta.get('split', 'N/A')}")
    snippet = query_text[:300] + ("..." if len(query_text) > 300 else "")
    print(f"  text        : {snippet}")

    print("\n[self-check] 使用上述文本作为查询：")
    run_query(
        query_text=query_text,
        model=model,
        collection=collection,
        n_results=n_results,
        where_filter=None,
    )


def main() -> None:
    """
    命令行入口：提供简单菜单进行质量验证。
    """
    parser = argparse.ArgumentParser(
        description="对 PubMed RCT BGE + Chroma 向量索引进行质量验证。"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="./chroma_medrag_bge",
        help="Chroma 持久化目录。",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="pubmed_rct_bge",
        help="Chroma 集合名称。",
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=5,
        help="每次查询返回的结果数量。",
    )

    args = parser.parse_args()

    model = load_bge_model("BAAI/bge-small-en-v1.5")
    collection = get_chroma_collection(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    while True:
        print("\n========== 查询菜单 ==========")
        print("1) 自检：随机抽取索引中文本作为查询")
        print("2) 手动查询：输入英文医学问题或句子")
        print("0) 退出")
        choice = input("请选择操作：").strip()

        if choice == "1":
            self_check(
                model=model,
                collection=collection,
                n_results=args.n_results,
            )
        elif choice == "2":
            q = input(
                "\n请输入英文查询文本（直接回车返回菜单）：\n> "
            ).strip()
            if not q:
                continue
            run_query(
                query_text=q,
                model=model,
                collection=collection,
                n_results=args.n_results,
                where_filter=None,
            )
        elif choice == "0" or choice == "":
            print("[exit] 结束查询。")
            break
        else:
            print("[menu] 无效选项，请重新输入。")


if __name__ == "__main__":
    main()
