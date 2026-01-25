# PubMed RCT 向量化与向量索引构建说明  
文件：`2026-01-25-doc-vector-index-pubmed-rct.md`  
工程目录：`medrag-rag/`

本说明基于上一阶段  
《2026-01-16-doc-chunking-pubmed-rct》（文档解析与分割），  
记录本阶段使用 BGE 嵌入模型与 Chroma 向量数据库，对 PubMed RCT 文本块进行向量化和索引构建的过程，以及基础的质量验证结果与潜在问题分析。

---

## 1. 目标与总体设计

**目标：**

在已完成的文本块数据基础上，为后续医学 RAG 系统提供一个可持久化、可检索的向量索引，支持：

- 按语义相似度检索相关的 PubMed RCT 文本块；
- 通过元数据（`doc_id`、`chunk_index`、`split` 等）追溯原始文献；
- 为后续本地 LLM（如 Qwen + RAG）提供高质量检索上下文。

**总体思路：**

1. 使用 BGE 英文嵌入模型，将每个文本块编码为定长向量；
2. 使用 Chroma 构建持久化向量索引，保存：
   - 向量（embeddings）
   - 文本内容（documents）
   - 元数据（metadatas：`doc_id`、`chunk_index` 等）
3. 输出索引统计信息（`stats` 文件），记录当前索引状态；
4. 通过自检与实际医学查询，对索引进行基础质量验证。

---

## 2. 嵌入模型配置

### 2.1 模型选择

本阶段选择的嵌入模型为：

- **模型名称**：`BAAI/bge-small-en-v1.5`
- **类型**：英文检索向量模型（small 版本）
- **向量维度**：384（由 `get_sentence_embedding_dimension()` 实际输出）

选择原因：

1. PubMed RCT 摘要为英文，与 BGE English 系列模型自然匹配；
2. `small` 版本参数量较小，在当前本地硬件环境下推理速度和稳定性较好；
3. BGE 系列针对“信息检索”任务优化，官方推荐为 query / passage 添加前缀指令，有利于提升语义检索效果。

### 2.2 使用方式

在向量化阶段，通过 `SentenceTransformer` 加载模型：

- **文本块（passage）嵌入**：

  - 输入形式：  
    `Represent this passage for retrieval: {text}`
  - 调用示例：
    ```python
    passages = [f"Represent this passage for retrieval: {t}" for t in texts]
    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    ```

- **查询（query）嵌入**（在质量验证脚本中使用）：

  - 输入形式：  
    `Represent this query for retrieval: {query_text}`
  - 使用同样的 encode 接口，并保持 `normalize_embeddings=True`，以便使用余弦相似度。

---

## 3. 向量索引构建流程

### 3.1 输入数据

输入文件来自上一阶段的 chunking 结果（本阶段仅使用 train 集）：

- 路径：`data/pubmed_rct_train_chunks.csv`
- 粒度：**每行一个文本块**
- 关键字段：
  - `chunk_id`
  - `text`
  - `doc_id`
  - `chunk_index`
  - `total_chunks`
  - `source_title`
  - `token_count`
  - `split`（本阶段为 `"train"`）

`validation` / `test` 集尚未纳入本次索引，可在后续迭代中按相同流程扩展。

### 3.2 构建脚本与调用方式

主脚本：

- 路径：`src/build_bge_index.py`

核心步骤：

1. **加载文本块数据**

   ```python
   chunks_df = load_chunks(splits=["train"])
   # 输出示例：
   # [data] 文本块数量：15000
   # [data] 列名：['chunk_id', 'text', 'doc_id', 'chunk_index',
   #               'total_chunks', 'source_title', 'token_count', 'split']