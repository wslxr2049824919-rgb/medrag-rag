# PubMed RCT 文档解析与分割说明

## 0. 背景与目标

本说明文档记录了在 `armanc/pubmed-rct20k` 数据集上所做的  
**文档解析与分割工作**，主要包括：

- 将原本“句子级”的 PubMed RCT 数据聚合为“摘要级文档”；
- 按既定策略将每篇文献映射为适合向量化和检索的文本单元（chunk）；
- 对生成的文本块数据集进行统计分析和质量验证；
- 将处理过程与结果以文档与数据文件的形式沉淀，方便后续 RAG 系统使用与复现。

本阶段的目标是为后续的 **向量化（embeddings）与 RAG 检索**  
提供一个 **结构清晰、质量可控的文本块数据集**，而不是追求最复杂的分割策略。  
在 PubMed RCT 这种“摘要长度适中、结构相对统一”的场景下，  
优先验证最简单、最稳定的方案（摘要级整体不分割）。

---

## 1. 原始数据与环境

### 1.1 数据来源与结构

- 数据来源：HuggingFace 数据集  
  `armanc/pubmed-rct20k`
- 使用的 split：
  - `train`
  - `validation`
  - `test`

原始数据为 **句子级 RCT 摘要**，每一行对应一条句子，字段包括：

- `abstract_id`：摘要 ID（可视作一篇文献的 ID）
- `sentence_id`：句子在摘要中的顺序索引
- `text`：句子内容（英文）
- `label`：该句子的语义类别  
  (`background` / `objective` / `methods` / `results` / `conclusions`)

即：原始数据并不是“一行一篇摘要”，而是“一行一句话”。

### 1.2 加载与 DataFrame 转换

数据通过 `datasets` 库加载，例如：

```python
from datasets import load_dataset

ds = load_dataset("armanc/pubmed-rct20k")
train = ds["train"]
df = train.to_pandas()
