"""
rag_chat.py

功能：
- 复用已经构建好的 Chroma 向量索引 ./chroma_medrag_mvp
- 使用 nomic-embed-text 作为检索阶段的向量模型（通过 Ollama）
- 使用 qwen3:8b 作为生成回答的 LLM（通过 Ollama）
- 在终端提供一个简单的命令行 RAG 医学问答界面
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


def build_rag_pipeline():
    """
    构建 RAG 所需的核心组件：
    1. Embeddings：把文本转换为向量
    2. VectorStore + Retriever：向量库 + 检索接口
    3. LLM：根据检索结果生成最终回答
    """

    # 1) 初始化 embedding 模型：nomic-embed-text（通过 Ollama）
    emb = OllamaEmbeddings(model="nomic-embed-text")

    # 2) 从本地目录加载已经持久化好的 Chroma 向量索引
    vs = Chroma(
        embedding_function=emb,
        persist_directory="./chroma_medrag_mvp",
    )

    # 3) 将向量库包装成 retriever，只关心“问题 -> top-k 文档”
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    # 4) 准备本地 LLM，这里使用 qwen3:8b（通过 Ollama）
    llm = Ollama(model="qwen3:8b")

    return retriever, llm


def rag_answer(question: str, retriever, llm, debug: bool = False) -> str:
    """
    最小 RAG 问答流程：
    1. 用 retriever.invoke(question) 检索相关文献片段
    2. 将检索到的片段拼接成一个 context
    3. 构造带“不要瞎编”的 prompt
    4. 调用 qwen3:8b 生成中文回答

    参数：
        question: 用户问题（中英文都可以）
        retriever: Chroma 提供的检索器
        llm: 本地 LLM（qwen3:8b）
        debug: 为 True 时会打印检索到的文献片段，方便调试
    """

    # 1) 检索阶段：从向量库中找到最相关的若干 Document
    docs = retriever.invoke(question)

    # 如果需要调试，可以打印出检索到的文献片段
    if debug:
        print("\n====== 检索到的文献片段（Top-3） ======")
        for i, d in enumerate(docs, 1):
            print(f"\n--- 片段 {i} ---")
            print(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))

    # 2) 将多个文献片段拼接成一个大的上下文字符串
    #    中间用分隔线方便模型区分不同片段
    context = "\n\n--- 文献片段 ---\n\n".join(
        d.page_content for d in docs
    )

    # 3) 构造 prompt，对模型明确说明角色和约束
    prompt = f"""
你是一名严谨的医学助手，回答问题时必须主要基于给定的文献片段，
不要编造文献中没有的信息。如果文献中没有提到，就坦诚说明不确定。

用户问题：
{question}

以下是从医学文献中检索到的相关片段（英文原文）：
{context}

请你：
1. 先用中文总结这些片段中与问题直接相关的关键信息；
2. 再给出一个简短的中文回答（可以适当翻译和解释专业词汇）。
    """.strip()

    # 4) 调用本地 LLM 生成回答
    answer = llm.invoke(prompt)
    return answer


def main():
    # 程序启动时只初始化一次 RAG 管道，后续循环复用 retriever 和 llm
    print("[medRAG] 正在初始化 RAG 管道（加载向量库和模型）...")
    retriever, llm = build_rag_pipeline()
    print("[medRAG] 初始化完成，可以开始提问。\n")

    # 简单的命令行交互循环
    while True:
        try:
            q = input("[medRAG] 请输入问题（直接回车退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[medRAG] 已退出。")
            break

        if not q:
            print("[medRAG] 空问题，退出。")
            break

        # 如果想看检索到的文献片段，可以把 debug 改成 True
        answer = rag_answer(q, retriever, llm, debug=False)

        print("\n[回答]")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()