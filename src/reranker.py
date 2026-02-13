"""
reranker.py

使用 BAAI/bge-reranker-base 对初步检索结果进行重排序：
- 先用 cross-encoder 计算 query 与候选文献片段的相关性得分（relevance）
- 再结合文献年份（recency）与期刊权威性（authority）做加权融合
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from hybrid_retriever import HybridResult


@dataclass
class RerankScore:
    """保存单个候选文献的各项评分与最终综合得分。"""
    relevance: float
    recency: float
    authority: float
    final_score: float


class BgeReranker:
    """
    使用 BAAI/bge-reranker-base 实现的重排序器。

    - relevance：由模型输出的相关性分数，经归一化处理
    - recency：根据 pub_year 计算出的时间新旧得分
    - authority：根据期刊名称估算的权威性得分
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        criteria_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            model_name: reranker 模型名称
            device: 使用的设备（"cuda" / "cpu"），默认自动检测
            criteria_weights: 多准则加权权重
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[reranker] 加载模型：{model_name}，device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 多准则权重，默认为题目给出的示例
        self.criteria_weights = criteria_weights or {
            "relevance": 0.6,
            "recency": 0.25,
            "authority": 0.15,
        }

    # ---------- 公共入口 ----------

    def rerank(
        self,
        query: str,
        candidates: List[HybridResult],
        top_k: Optional[int] = None,
    ) -> List[HybridResult]:
        """
        对候选文献列表进行重排序。

        Args:
            query: 原始查询字符串（建议使用 cleaned_query）
            candidates: 多路检索输出的候选列表（HybridResult）
            top_k: 只返回前 top_k 条（不传则返回全部）

        Returns:
            重新排序后的 HybridResult 列表（fused_score 已更新为综合得分）
        """
        if not candidates:
            return []

        # 1. 计算模型相关性得分
        relevance_scores = self._compute_relevance_scores(query, candidates)

        # 2. 计算 recency / authority 得分
        recency_scores = [self._compute_recency_score(c) for c in candidates]
        authority_scores = [self._compute_authority_score(c) for c in candidates]

        # 3. 对三个分量做归一化（0~1）
        rel_norm = self._min_max_normalize(relevance_scores)
        rec_norm = self._min_max_normalize(recency_scores)
        auth_norm = self._min_max_normalize(authority_scores)

        w_rel = self.criteria_weights.get("relevance", 0.6)
        w_rec = self.criteria_weights.get("recency", 0.25)
        w_auth = self.criteria_weights.get("authority", 0.15)

        reranked: List[HybridResult] = []

        for idx, c in enumerate(candidates):
            r = rel_norm[idx]
            t = rec_norm[idx]
            a = auth_norm[idx]

            final_score = w_rel * r + w_rec * t + w_auth * a

            # 将各项得分写回 HybridResult，方便观察
            if "rerank" not in c.scores:
                c.scores["rerank"] = final_score
            else:
                c.scores["rerank"] = final_score

            c.metadata.setdefault("rerank_detail", {})
            c.metadata["rerank_detail"].update(
                {
                    "relevance_raw": float(relevance_scores[idx]),
                    "relevance_norm": float(r),
                    "recency_score": float(t),
                    "authority_score": float(a),
                    "weights": {
                        "relevance": w_rel,
                        "recency": w_rec,
                        "authority": w_auth,
                    },
                }
            )

            # 更新 fused_score 为综合得分
            c.fused_score = final_score

            reranked.append(c)

        # 按综合得分从高到低排序
        reranked.sort(key=lambda x: x.fused_score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    # ---------- 内部工具：模型相关性 ----------

    def _compute_relevance_scores(
        self,
        query: str,
        candidates: List[HybridResult],
        max_length: int = 512,
    ) -> List[float]:
        """
        使用 bge-reranker 模型计算每个候选与 query 的相关性原始分数。
        """
        texts = [c.text for c in candidates]
        # 模型输入是 (query, document) 对
        pairs = list(zip([query] * len(texts), texts))

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            # bge-reranker-base 是一个二分类 / 回归头，这里取 logits 的单值即可
            logits = outputs.logits  # [batch_size, 1] 或 [batch_size]
            scores = logits.squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        return scores

    # ---------- 内部工具：recency / authority ----------

    def _compute_recency_score(self, candidate: HybridResult) -> float:
        """
        根据文献年份简单估算时间新旧得分：
        - 如果 metadata 中包含 'pub_year' 或 'year' 字段，则按年份计算
        - 否则返回一个中性值 0.5
        """
        meta = candidate.metadata or {}
        year = meta.get("pub_year") or meta.get("year")

        try:
            year = int(year)
        except Exception:
            # 无法解析年份时返回中性分
            return 0.5

        # 这里假设现在时间约为 2025 年，简单线性衰减：
        # 最近 0~5 年：得分逐渐接近 1
        # 5~15 年：逐渐降到 0
        current_year = 2025
        delta = current_year - year

        if delta <= 0:
            return 1.0
        elif delta <= 5:
            # 0~5 年内线性从 1 降到 0.7
            return 1.0 - 0.06 * delta
        elif delta <= 15:
            # 5~15 年内线性从 0.7 降到 0
            return max(0.0, 0.7 - 0.07 * (delta - 5))
        else:
            return 0.0

    def _compute_authority_score(self, candidate: HybridResult) -> float:
        """
        根据期刊名称估算权威性得分：
        - 简单使用一个小字典示例
        - 如果没有期刊信息，返回中性值 0.5
        """
        meta = candidate.metadata or {}
        journal = str(meta.get("journal", "")).lower().strip()

        if not journal:
            return 0.5

        # 示例权重配置，可根据需要扩展 / 替换
        high_tier = [
            "the new england journal of medicine",
            "new england journal of medicine",
            "lancet",
            "jama",
            "bmj",
            "nature",
        ]
        mid_tier = [
            "circulation",
            "stroke",
            "diabetes care",
            "journal of the american college of cardiology",
        ]

        if any(j in journal for j in high_tier):
            return 1.0
        if any(j in journal for j in mid_tier):
            return 0.8

        # 其他期刊给一个基础分
        return 0.6

    # ---------- 内部工具：归一化 ----------

    @staticmethod
    def _min_max_normalize(values: List[float]) -> List[float]:
        """
        将一组数线性归一化到 [0, 1]。
        如果所有值相同，则全部返回 0.5。
        """
        if not values:
            return []

        v_min = min(values)
        v_max = max(values)

        if math.isclose(v_min, v_max):
            return [0.5 for _ in values]

        return [(v - v_min) / (v_max - v_min) for v in values]


# ---------- 简单 demo：仅测试 reranker 对候选结果的重新排序 ----------

def main():
    from hybrid_retriever import MultiPathRetriever
    from query_understanding import MedicalQueryProcessor
    from keyword_bm25 import BM25KeywordIndex
    from query_bge_index import load_bge_model, get_chroma_collection

    print("[reranker-demo] 加载 BGE 模型和 Chroma 集合...")
    model = load_bge_model("BAAI/bge-small-en-v1.5")
    collection = get_chroma_collection()

    print("[reranker-demo] 加载 BM25 索引...")
    bm25_index = BM25KeywordIndex()
    bm25_index.build()

    print("[reranker-demo] 初始化查询理解模块与多路检索器...")
    processor = MedicalQueryProcessor()
    retriever = MultiPathRetriever(
        collection=collection,
        embedding_model=model,
        bm25_index=bm25_index,
    )

    print("[reranker-demo] 初始化 reranker...")
    reranker = BgeReranker()

    print("[reranker-demo] 初始化完成，可以开始测试重排序。\n")

    while True:
        q = input("[reranker-demo] 请输入医学问题（回车退出）：").strip()
        if not q:
            print("[reranker-demo] 退出。")
            break

        analysis = processor.process(q)

        # 先用多路检索得到候选，数量可以稍微多一些
        candidates = retriever.retrieve(
            query_info=analysis,
            top_k_vector=10,
            top_k_keyword=10,
            fusion_strategy="rrf",
        )

        print(f"\n[reranker-demo] 候选数量：{len(candidates)}，开始重排序...")

        reranked = reranker.rerank(
            query=analysis.cleaned_query,
            candidates=candidates,
            top_k=5,  # 只看前 5 条
        )

        print(f"[reranker-demo] 重排序后 Top {len(reranked)}：")
        for rank, d in enumerate(reranked, start=1):
            detail = d.metadata.get("rerank_detail", {})
            print(f"\n== Rank {rank} ==")
            print(f"doc_id   : {d.doc_id}")
            print(f"chunk_id : {d.chunk_id}")
            print(f"final    : {d.fused_score:.4f}")
            print(f"relevance_norm: {detail.get('relevance_norm')}")
            print(f"recency_score : {detail.get('recency_score')}")
            print(f"authority_score: {detail.get('authority_score')}")
            print(f"text     : {d.text[:200]}...")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
