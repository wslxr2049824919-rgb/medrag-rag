"""
build_index.py

功能：
- 从 HuggingFace 加载 PubMed RCT 数据集（armanc/pubmed-rct20k）
- 取 train 集前 500 条样本，作为 RAG MVP 的索引数据
- 使用 Ollama 的 nomic-embed-text 生成向量
- 使用 Chroma 构建向量索引，并持久化到 ./chroma_medrag_mvp

后续使用：
- 先运行本脚本构建索引
- 然后再用 rag_chat.py 做问答
"""

from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document


def main():
    # 1. 加载数据集
    print("[build_index] 正在加载数据集 armanc/pubmed-rct20k ...")
    ds = load_dataset("armanc/pubmed-rct20k")

    train_split = ds["train"]
    # 这里只取前 500 条做 MVP 验证，后面要扩展可以改这个数
    subset = train_split.select(range(500))

    # 2. 将样本转换为 LangChain 的 Document 列表
    print("[build_index] 正在从子集构建 Document 列表 ...")
    docs = []
    for row in subset:
        text = row.get("text", "")
        if not text:
            # 如果这一条 text 为空，直接跳过
            continue
        # 这里只用 text 作为检索内容，metadata 暂时不加
        docs.append(Document(page_content=str(text)))

    print(f"[build_index] Document 数量: {len(docs)}")

    # 3. 初始化 embedding 模型（通过 Ollama 调用 nomic-embed-text）
    print("[build_index] 正在初始化 embedding 模型: nomic-embed-text ...")
    emb = OllamaEmbeddings(model="nomic-embed-text")

    # 4. 创建 Chroma 向量库，并持久化到本地目录
    print("[build_index] 正在创建 Chroma 向量索引 ...")
    Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory="./chroma_medrag_mvp",
    )

    print("[build_index] 索引构建完成，已持久化到 ./chroma_medrag_mvp")


if __name__ == "__main__":
    main()