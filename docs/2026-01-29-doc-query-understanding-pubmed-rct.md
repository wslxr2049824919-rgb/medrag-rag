PubMed RCT 查询理解与增强模块说明

文件：2026-01-29-doc-query-understanding-pubmed-rct.md
工程目录：medrag-rag/

本阶段工作基于前两周成果：
	•	2026-01-16-doc-chunking-pubmed-rct（文档解析与分割）
	•	2026-01-25-doc-vector-index-pubmed-rct（向量化与向量索引构建）

在已有的 PubMed RCT 文本块 + BGE 向量索引 + Chroma 的基础上，增加了一层查询理解与增强模块，并接入到检索脚本中，用于支持后续医学 RAG 问答系统的智能检索。

⸻

1. 本阶段目标（干了什么）
	•	不是只拿用户原始问题去算 embedding，而是先“理解问题再检索”；
	•	让系统认识常见医学缩写、口语化说法（如 mi, heart attack）；
	•	自动扩展出标准术语（如 myocardial infarction）；
	•	为 BGE 构造更合适的查询文本（带推荐的指令前缀）；
	•	预留关键词检索和时间过滤的结构，方便以后接 BM25、时间过滤等。

一句话：先把“你问的问题”变成“模型更容易理解的问题”，再去查向量库。

⸻

2. 查询理解模块（src/query_understanding.py）

本模块主要包含三块内容：
	1.	静态医学同义词词典 MEDICAL_SYNONYMS
示例：
MEDICAL_SYNONYMS = {
    "mi": ["myocardial infarction", "heart attack"],
    "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes"],
    "heart attack": ["myocardial infarction"],
    "high blood pressure": ["hypertension"],
}
用途：当用户问句里出现这些缩写或口语化表达时，自动扩展出更规范的术语。

	2.	医学实体正则 MEDICAL_PATTERNS
示例：
MEDICAL_PATTERNS = {
    "drug": r"\b(aspirin|metformin|atorvastatin|warfarin|insulin)\b",
    "disease": r"\b(myocardial infarction|heart attack|stroke|diabetes|hypertension|knee osteoarthritis)\b",
}
用途：从查询中大致识别“药物名 / 疾病名”等简单实体。

	3.	核心类 MedicalQueryProcessor
	•	输入：原始查询字符串 query
	•	输出：一个结构化结果 QueryAnalysisResult，包含：
	•	raw_query: 原始问题
	•	cleaned_query: 去掉末尾标点、小写化后的问题
	•	entities: 识别出的实体，如 {"drug": ["metformin"]}
	•	expanded_terms: 同义词扩展出的术语
	•	vector_queries: 为 BGE 准备好的向量查询文本（一个或多个版本）
	•	keyword_query: 为关键词检索准备的查询字符串
	•	filters: 简单过滤条件（例如 {"time_window_years": 5}）
内部做了几件事：
	•	清洗查询（去空格、统一小写、去掉末尾 ?/.）
	•	用正则识别药物、疾病等实体；
	•	用 MEDICAL_SYNONYMS 扩展缩写和口语化说法；
	•	识别简单时间表达（如 in the last 5 years → time_window_years = 5）；
	•	按 BGE 最佳实践构造向量查询文本，例如：
	•	基础版：
Represent this question for searching relevant passages: short-term effects of metformin on mi in the last 5 years
	•	增强版（带同义词提示）：
Represent this question for searching relevant passages: ... Consider related terms: heart attack, myocardial infarction.

⸻

3. 与向量检索的集成（src/query_bge_index.py）

在上一阶段，query_bge_index.py 只做了：
	•	加载 BGE 嵌入模型（如 BAAI/bge-small-en-v1.5）；
	•	加载 Chroma 集合 pubmed_rct_bge；
	•	自检 + 简单手动查询。

本阶段在此基础上增加：
	1.	引入查询处理器：
from query_understanding import MedicalQueryProcessor, QueryAnalysisResult

processor = MedicalQueryProcessor()
2.	手动查询时，先做查询分析，再检索：
	•	用户输入问题 q
	•	调用 analysis = processor.process(q)
	•	打印：
	•	cleaned query
	•	entities
	•	expanded_terms
	•	filters
	•	vector_queries 列表
	•	选用 analysis.vector_queries 中最后一个（通常是带同义词提示的增强版）作为最终向量查询文本，送入 BGE + Chroma 进行检索。
	3.	embed_query() 不再重复添加 BGE 指令前缀，只负责对传入文本做 encode()。

⸻

4. 示例行为（实际跑过的例子）

示例 1：英文 + 缩写 + 时间范围

输入查询：

short-term effects of metformin on mi in the last 5 years?

查询理解模块给出的结果大致为：
	•	entities: {"drug": ["metformin"]}
	•	expanded_terms: ["heart attack", "myocardial infarction"]
	•	filters: {"time_window_years": 5}
	•	vector_queries 中包含增强版文本（带 “Consider related terms” 提示）

在此基础上做向量检索，返回的 Top 文献片段主要是：
	•	metformin 对心肌梗死后预后、左心室功能等的影响；
	•	metformin 在心血管结局试验中的作用；
	•	metformin 对炎症标志物、凝血指标等的影响。

说明：系统成功把 “mi” 理解为 myocardial infarction/heart attack，并在检索时利用了这些信息。

示例 2：药物 + 疾病

输入：

what is the effect of aspirin on stroke?

	•	entities 中识别出 {"drug": ["aspirin"], "disease": ["stroke"]}
	•	没有额外同义词扩展时，仍然构造出合规的 BGE 查询文本，可以正常检索相关试验。

⸻

5. 已知限制与后续方向
	•	静态词典和正则目前只覆盖了非常有限的一小部分药物 / 疾病 / 缩写，主要用于示例；
	•	目前增强逻辑只对英文有效，中文查询只做了基础清洗；
	•	时间过滤条件（filters）尚未接入 Chroma 的 where 过滤；
	•	keyword_query 暂未真正接入 BM25 / 倒排索引，未来可考虑做 hybrid 检索。

后续可以考虑：
	•	从 UMLS / MeSH 或开源医学词库中自动构建更大的同义词词典；
	•	对 chunk 元数据增加 pub_date，使时间过滤真正生效；
	•	引入关键词检索引擎（如 Elasticsearch）与向量检索结合。

⸻

6. 本阶段产出
	•	代码：
	•	src/query_understanding.py：医学查询理解与增强模块；
	•	src/query_bge_index.py：接入查询增强后的向量检索脚本（保留自检功能）。
	•	文档：
	•	docs/2026-01-29-doc-query-understanding-pubmed-rct.md（本文件）。

