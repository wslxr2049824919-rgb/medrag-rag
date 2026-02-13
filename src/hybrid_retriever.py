"""
hybrid_retriever.py

多路检索（向量 + BM25）与结果融合：
- 向量检索：BGE + Chroma
- 关键词检索：BM25KeywordIndex
- 融合策略：simple / rrf / weighted

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

from query_understanding import MedicalQueryProcessor, QueryAnalysisResult
from keyword_bm25 import BM25KeywordIndex, BM25Document

from query_bge_index import (
    load_bge_model,
    get_chroma_collection,
    embed_query,
)


@dataclass
class HybridResult:
    """融合后的检索结果，统一返回格式。"""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    scores: Dict[str, float]     # 各通路得分，例如 {"vector": 0.9, "bm25": 0.4}
    fused_score: float           # 最终融合得分


class MultiPathRetriever:
    """
    多路检索器：
    - 使用向量检索（BGE + Chroma）
    - 使用关键词检索（BM25）
    - 提供多种结果融合策略
    """

    def __init__(
        self,
        collection: Collection,
        embedding_model: SentenceTransformer,
        bm25_index: BM25KeywordIndex,
    ):
        self.collection = collection
        self.embedding_model = embedding_model
        self.bm25_index = bm25_index

    # ---------------- 向量检索 ----------------

    def _vector_search(self, query_text: str, top_k: int) -> List[HybridResult]:
        """
        使用 Chroma 向量检索，返回统一的 HybridResult 列表。
        这里将距离 distance 转换为一个简单的“相似度分数”。
        """
        # 将 query 转换为嵌入向量，使用之前的 embed_query 工具函数
        query_embedding = embed_query(query_text, self.embedding_model)

        # Chroma 检索
        res = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs: List[HybridResult] = []

        # Chroma 返回的是按 query 批次的列表，这里只用第一个 query 的结果
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        for doc_text, meta, dist in zip(documents, metadatas, distances):
            # Chroma 是“距离越小越相似”，这里简单转成“相似度”
            sim = 1.0 / (1.0 + float(dist))

            chunk_id = str(meta.get("chunk_id", meta.get("id", "")))
            doc_id = str(meta.get("doc_id", ""))

            hr = HybridResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=doc_text,
                metadata=meta,
                scores={"vector": sim},
                fused_score=sim,  # 先暂存，后续融合会覆盖
            )
            docs.append(hr)

        return docs

    # ---------------- BM25 关键词检索 ----------------

    def _keyword_search(self, query_text: str, top_k: int) -> List[HybridResult]:
        """
        使用 BM25 进行关键词检索，转换为 HybridResult 列表。
        这里暂时不直接使用 BM25 原始分数，而是后面基于“名次”做融合。
        """
        bm25_docs: List[BM25Document] = self.bm25_index.search(query_text, top_k=top_k)

        results: List[HybridResult] = []
        for d in bm25_docs:
            hr = HybridResult(
                chunk_id=d.chunk_id,
                doc_id=d.doc_id,
                text=d.text,
                metadata=d.metadata,
                scores={"bm25": 1.0},  # 先放一个占位，融合时会用名次信息
                fused_score=1.0,
            )
            results.append(hr)

        return results

    # ---------------- 结果融合 ----------------

    def _fuse_results(
        self,
        vector_results: List[HybridResult],
        keyword_results: List[HybridResult],
        fusion_strategy: str = "rrf",
        weight_vector: float = 0.7,
        weight_keyword: float = 0.3,
    ) -> List[HybridResult]:
        """
        根据指定策略融合两路结果。
        支持：
        - simple: 向量检索结果在前，BM25 结果补充，去重
        - rrf: Reciprocal Rank Fusion（按 1/(k + rank) 融合）
        - weighted: 按 1/rank 做简单加权，向量权重更高
        """
        fusion_strategy = fusion_strategy.lower()
        # 使用 chunk_id 作为统一的“文档标识”
        merged: Dict[str, HybridResult] = {}

        # 记录名次
        vector_rank: Dict[str, int] = {}
        keyword_rank: Dict[str, int] = {}

        # 向量路：记录 rank（从 1 开始）
        for idx, hr in enumerate(vector_results, start=1):
            cid = hr.chunk_id or f"vec_{idx}"
            vector_rank[cid] = idx
            if cid not in merged:
                merged[cid] = hr
            else:
                merged[cid].scores.update(hr.scores)

        # BM25 路：记录 rank
        for idx, hr in enumerate(keyword_results, start=1):
            cid = hr.chunk_id or f"kw_{idx}"
            keyword_rank[cid] = idx
            if cid not in merged:
                merged[cid] = hr
            else:
                merged[cid].scores.update(hr.scores)

        # 按不同策略计算 fused_score
        for cid, hr in merged.items():
            vr = vector_rank.get(cid)
            kr = keyword_rank.get(cid)

            if fusion_strategy == "simple":
                # simple 模式下 fused_score 不重要，后面按“向量优先 + BM25 补充”的顺序排序
                hr.fused_score = 0.0

            elif fusion_strategy == "rrf":
                # Reciprocal Rank Fusion: score = sum(1 / (k + rank))
                # 这里选 k = 60（常见经验值）
                k_const = 60.0
                score = 0.0
                if vr is not None:
                    score += 1.0 / (k_const + vr)
                if kr is not None:
                    score += 1.0 / (k_const + kr)
                hr.fused_score = score

            elif fusion_strategy == "weighted":
                # 简单基于 rank 的加权评分：score = w_vec * (1/vr) + w_kw * (1/kr)
                score = 0.0
                if vr is not None:
                    score += weight_vector * (1.0 / float(vr))
                if kr is not None:
                    score += weight_keyword * (1.0 / float(kr))
                hr.fused_score = score

            else:
                # 未知策略时退回到 rrf
                k_const = 60.0
                score = 0.0
                if vr is not None:
                    score += 1.0 / (k_const + vr)
                if kr is not None:
                    score += 1.0 / (k_const + kr)
                hr.fused_score = score

        # 根据策略排序
        results = list(merged.values())

        if fusion_strategy == "simple":
            # simple：向量结果在前，保持各自内部顺序，BM25 只补充没出现过的
            ordered: List[HybridResult] = []
            seen: set[str] = set()

            for hr in vector_results:
                cid = hr.chunk_id
                if cid not in seen:
                    ordered.append(merged[cid])
                    seen.add(cid)

            for hr in keyword_results:
                cid = hr.chunk_id
                if cid not in seen:
                    ordered.append(merged[cid])
                    seen.add(cid)

            return ordered

        else:
            # 其它策略：按 fused_score 从大到小排序
            results.sort(key=lambda x: x.fused_score, reverse=True)
            return results

    # ---------------- 对外主接口 ----------------

    def retrieve(
        self,
        query_info: QueryAnalysisResult,
        top_k_vector: int = 5,
        top_k_keyword: int = 5,
        fusion_strategy: str = "rrf",
    ) -> List[HybridResult]:
        """
        使用查询理解模块的输出结果，执行多路检索并融合。

        Args:
            query_info: 查询理解模块的输出
            top_k_vector: 向量检索数量
            top_k_keyword: 关键词检索数量
            fusion_strategy: 'simple' / 'rrf' / 'weighted'

        Returns:
            融合后的 HybridResult 列表
        """
        # 1) 准备向量检索用的查询文本（优先使用带同义词提示的版本）
        if query_info.vector_queries:
            vector_query_text = query_info.vector_queries[-1]
        else:
            vector_query_text = query_info.cleaned_query

        # 2) 准备关键词检索用的查询文本（若有 keyword_query 优先使用）
        keyword_query_text = query_info.keyword_query or query_info.cleaned_query

        # 3) 分别检索
        vector_results = self._vector_search(vector_query_text, top_k=top_k_vector)
        keyword_results = self._keyword_search(keyword_query_text, top_k=top_k_keyword)

        # 4) 融合
        fused_results = self._fuse_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            fusion_strategy=fusion_strategy,
        )

        return fused_results


# ---------------- 简单 demo：从命令行测试多路检索 ----------------

def main():
    print("[hybrid] 加载 BGE 模型和 Chroma 集合...")
    model = load_bge_model("BAAI/bge-small-en-v1.5")
    collection = get_chroma_collection()  # 假设内部已经配置好 collection_name / persist_directory

    print("[hybrid] 加载 BM25 索引...")
    bm25_index = BM25KeywordIndex()
    bm25_index.build()

    print("[hybrid] 初始化查询理解模块...")
    processor = MedicalQueryProcessor()

    retriever = MultiPathRetriever(
        collection=collection,
        embedding_model=model,
        bm25_index=bm25_index,
    )

    print("[hybrid] 初始化完成，可以开始测试多路检索。\n")

    while True:
        q = input("[hybrid] 请输入医学问题（回车退出）：").strip()
        if not q:
            print("[hybrid] 退出。")
            break

        # 1) 查询理解与增强
        analysis: QueryAnalysisResult = processor.process(q)

        print("\n[analysis] 清洗后的查询：", analysis.cleaned_query)
        print("[analysis] 实体识别：", analysis.entities)
        print("[analysis] 同义词扩展：", analysis.expanded_terms)
        print("[analysis] 过滤条件：", analysis.filters)

        # 2) 多路检索 + 融合
        fused = retriever.retrieve(
            query_info=analysis,
            top_k_vector=5,
            top_k_keyword=5,
            fusion_strategy="rrf",  # 也可以改成 'simple' 或 'weighted'
        )

        print(f"\n[hybrid] 融合后结果数量：{len(fused)}")
        for rank, d in enumerate(fused[:5], start=1):
            print(f"\n== Rank {rank} ==")
            print(f"chunk_id : {d.chunk_id}")
            print(f"doc_id   : {d.doc_id}")
            print(f"scores   : {d.scores}")
            print(f"fused    : {d.fused_score:.6f}")
            print(f"text     : {d.text[:200]}...")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
