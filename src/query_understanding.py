"""
query_understanding.py

负责对医学查询做“理解与增强”：
- 基础清洗（去空格、统一大小写等）
- 识别简单的医学实体（如常见药物）
- 根据静态同义词词典扩展查询
- 生成向量检索用的查询文本（BGE 前缀）
- 生成关键词检索用的查询文本（后续可对接 BM25 等）
- 提取简单过滤条件（如时间范围），目前为简化实现
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any


MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    # 缩写 -> 同义表达
    "mi": ["myocardial infarction", "heart attack"],
    "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes"],
    # 也可以添加“常用说法 -> 标准术语”
    "heart attack": ["myocardial infarction"],
    "high blood pressure": ["hypertension"],
}


# 简单医学实体模式：用于识别常见药物 / 疾病名等。
MEDICAL_PATTERNS: Dict[str, str] = {
    "drug": r"\b(aspirin|metformin|atorvastatin|warfarin|insulin)\b",
    "disease": r"\b(myocardial infarction|heart attack|stroke|diabetes|hypertension|knee osteoarthritis)\b",
}


@dataclass
class QueryAnalysisResult:
    """
    查询分析的结构化结果。
    """
    raw_query: str                              # 原始用户输入
    cleaned_query: str                          # 清洗后的查询（小写、去符号等）
    entities: Dict[str, List[str]] = field(default_factory=dict)   # 识别到的实体，按类型分类
    expanded_terms: List[str] = field(default_factory=list)        # 同义词扩展得到的术语列表
    vector_queries: List[str] = field(default_factory=list)        # 向量检索用的查询文本（可以有多个版本）
    keyword_query: str = ""                                        # 关键词检索用的查询字符串
    filters: Dict[str, Any] = field(default_factory=dict)          # 过滤条件（时间范围等）


class MedicalQueryProcessor:
    """
    医学查询理解与增强处理器。

    使用方式：
        processor = MedicalQueryProcessor()
        result = processor.process("short-term effects of metformin on cardiovascular disease")
    """

    def __init__(
        self,
        synonyms: Dict[str, List[str]] | None = None,
        patterns: Dict[str, str] | None = None,
    ) -> None:
        self.synonyms = synonyms or MEDICAL_SYNONYMS
        self.patterns = patterns or MEDICAL_PATTERNS

    # ===== 内部工具方法 =====

    def _clean_query(self, query: str) -> str:
        """
        基础清洗：
        - 去掉首尾空格
        - 统一为小写
        - 去掉句末问号等简单标点
        """
        q = query.strip()
        # 可以根据需要扩展更多清洗规则
        q = q.rstrip("？?!.")
        return q.lower()

    def _detect_entities(self, cleaned_query: str) -> Dict[str, List[str]]:
        """
        使用预定义的正则模式识别简单医学实体。
        返回按实体类型分类的字典，例如：
            {
                "drug": ["metformin"],
                "disease": ["hypertension"]
            }
        """
        entities: Dict[str, List[str]] = {}
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, cleaned_query)
            if matches:
                # 去重并保持大小写与 cleaned_query 一致（已经是小写）
                unique_matches = sorted(set(m.lower() for m in matches))
                entities[entity_type] = unique_matches
        return entities

    def _expand_synonyms(self, cleaned_query: str) -> List[str]:
        """
        根据静态同义词词典，对查询中的术语做简单扩展。
        示例：
            输入："mi"
            输出：["myocardial infarction", "heart attack"]
        实现方式：
            遍历 MEDICAL_SYNONYMS，检测键是否出现在查询中（按单词边界匹配）。
        """
        expanded: List[str] = []

        for key, syns in self.synonyms.items():
            pattern = r"\b" + re.escape(key.lower()) + r"\b"
            if re.search(pattern, cleaned_query):
                expanded.extend(syns)

        # 去重
        expanded = sorted(set(t.lower() for t in expanded))
        return expanded

    def _extract_filters(self, cleaned_query: str) -> Dict[str, Any]:
        """
        提取简单的过滤条件。
        当前仅实现一个示例：识别“in the last X years”模式，作为时间窗口。

        示例：
            "in the last 5 years" -> {"time_window_years": 5}
        """
        filters: Dict[str, Any] = {}

        # 匹配类似 "in the last 5 years" 的表达
        m = re.search(r"in the last (\d+)\s+years?", cleaned_query)
        if m:
            years = int(m.group(1))
            filters["time_window_years"] = years

        return filters

    def _build_vector_queries(
        self,
        cleaned_query: str,
        expanded_terms: List[str],
    ) -> List[str]:
        """
        构造向量检索用的查询文本。

        BGE 官方推荐为 query 添加指令前缀，这里采用：
            "Represent this question for searching relevant passages: {query}"

        同时，如果存在扩展术语，可以构造一个“带同义词提示”的版本。
        """
        vector_queries: List[str] = []

        # 基础版本：只用清洗后的原始查询
        base_query = (
            f"Represent this question for searching relevant passages: "
            f"{cleaned_query}"
        )
        vector_queries.append(base_query)

        # 扩展版本：将同义词以自然语言提示补充进去
        if expanded_terms:
            expansions = ", ".join(expanded_terms)
            expanded_query = (
                f"Represent this question for searching relevant passages: "
                f"{cleaned_query}. Consider related terms: {expansions}."
            )
            vector_queries.append(expanded_query)

        return vector_queries

    def _build_keyword_query(
        self,
        cleaned_query: str,
        entities: Dict[str, List[str]],
        expanded_terms: List[str],
    ) -> str:
        """
        构造一个简单的关键词检索查询字符串。

        思路：
            - 以 cleaned_query 为基础
            - 如果有识别到的实体和扩展同义词，用 OR 的方式附加
        示例输出：
            "metformin cardiovascular disease (myocardial infarction OR heart attack)"
        """
        parts: List[str] = [cleaned_query]

        # 将实体展平成一维列表
        entity_terms: List[str] = []
        for values in entities.values():
            entity_terms.extend(values)

        # 汇总所有“加权关键术语”
        all_terms = sorted(set(entity_terms + expanded_terms))

        if all_terms:
            or_block = " OR ".join(all_terms)
            parts.append(f"({or_block})")

        # 最终字符串
        keyword_query = " ".join(parts)
        return keyword_query

    # ===== 对外主入口 =====

    def process(self, query: str) -> QueryAnalysisResult:
        """
        对单条查询执行完整的“理解与增强”流程。
        """
        cleaned = self._clean_query(query)
        entities = self._detect_entities(cleaned)
        expanded_terms = self._expand_synonyms(cleaned)
        filters = self._extract_filters(cleaned)
        vector_queries = self._build_vector_queries(cleaned, expanded_terms)
        keyword_query = self._build_keyword_query(
            cleaned_query=cleaned,
            entities=entities,
            expanded_terms=expanded_terms,
        )

        return QueryAnalysisResult(
            raw_query=query,
            cleaned_query=cleaned,
            entities=entities,
            expanded_terms=expanded_terms,
            vector_queries=vector_queries,
            keyword_query=keyword_query,
            filters=filters,
        )


def _demo() -> None:
    
    processor = MedicalQueryProcessor()

    while True:
        try:
            q = input("\n请输入医学查询（回车退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] 结束 demo。")
            break

        if not q:
            print("[exit] 空输入，结束 demo。")
            break

        result = processor.process(q)

        print("\n[解析结果]")
        print(f"- 原始查询(raw): {result.raw_query}")
        print(f"- 清洗后(cleaned): {result.cleaned_query}")
        print(f"- 实体(entities): {result.entities}")
        print(f"- 同义词扩展(expanded_terms): {result.expanded_terms}")
        print(f"- 向量查询(vector_queries):")
        for i, vq in enumerate(result.vector_queries, 1):
            print(f"  [{i}] {vq}")
        print(f"- 关键词查询(keyword_query): {result.keyword_query}")
        print(f"- 过滤条件(filters): {result.filters}")


if __name__ == "__main__":
    _demo()
