# medRAG: 医学问答 RAG 系统（MVP）

本项目基于 PubMed RCT 数据集（`armanc/pubmed-rct20k`）搭建了一个
**句子级医学问答 RAG 系统原型（MVP）**，支持：

- 从真实医学文献中检索与问题相关的句子；
- 将检索结果作为上下文，调用本地大模型（如 `qwen3:8b`）生成中文回答；
- 对数据集的结构、文本长度分布、领域语言特性和分割策略进行了系统分析。

详细的数据分析与设计说明见：

> `docs/2026-01-10-rag-design.md`

---

## 环境与依赖

推荐使用 conda 管理虚拟环境。

```bash
# 进入项目根目录
cd ~/medrag-rag

# 激活已有环境（之前已创建）
conda activate medrag