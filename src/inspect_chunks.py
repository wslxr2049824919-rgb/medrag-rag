"""
inspect_chunks.py

功能：
- 读取 chunk_pubmed_rct.py 生成的文本块数据集
  data/pubmed_rct_train_chunks.csv
- 对文本块做基础质量检查和预览：
  1) 整体数量、字段信息
  2) token_count 分布（min/mean/p50/p90/p95/p99/max）
  3) 打印 token 最长的前若干条 chunk 片段
  4) 随机抽样若干条 chunk 做人工肉眼检查
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # 1) 读取文件
    chunks_path = Path("data/pubmed_rct_train_chunks.csv")

    if not chunks_path.exists():
        print(f"[error] 找不到文件：{chunks_path}")
        print("请先运行：python src/chunk_pubmed_rct.py 生成该文件。")
        return

    print(f"[load] 正在读取文本块数据集：{chunks_path}")
    df = pd.read_csv(chunks_path)

    # 2) 基本信息
    print("\n[basic] DataFrame 基本信息：")
    print(f"- 总行数（文本块数）：{len(df)}")
    print(f"- 字段列表：{list(df.columns)}")

    # 3) token_count 统计分布
    if "token_count" in df.columns:
        tc = df["token_count"].fillna(0)

        desc = {
            "min": tc.min(),
            "mean": tc.mean(),
            "p50": tc.quantile(0.5),
            "p90": tc.quantile(0.9),
            "p95": tc.quantile(0.95),
            "p99": tc.quantile(0.99),
            "max": tc.max(),
        }

        print("\n[stats] token_count 分布（按单词数近似 token）：")
        for k, v in desc.items():
            print(f"  {k:>4}: {v:.2f}")
    else:
        print("\n[stats] 未找到 token_count 字段，无法统计分布。")

    # 4) 查看 token 最长的前 5 条 chunk
    if "token_count" in df.columns:
        print("\n[longest] token_count 最大的前 5 条文本块：")
        df_long = df.sort_values("token_count", ascending=False).head(5)

        for idx, row in df_long.iterrows():
            print("\n--- 长文本块 ---")
            print(f"doc_id      : {row.get('doc_id')}")
            print(f"chunk_id    : {row.get('chunk_id')}")
            print(f"chunk_index : {row.get('chunk_index')}")
            print(f"total_chunks: {row.get('total_chunks')}")
            print(f"token_count : {row.get('token_count')}")
            text = str(row.get("text", ""))
            # 只打印前 400 个字符，避免刷屏
            preview = text[:400] + ("..." if len(text) > 400 else "")
            print(f"text preview: {preview}")
    else:
        print("\n[longest] 无法根据 token_count 排序。")

    # 5) 随机抽样若干条 chunk 人工检查
    n_sample = 5
    if len(df) <= n_sample:
        sample_df = df.copy()
    else:
        sample_df = df.sample(n=n_sample, random_state=42)

    print(f"\n[sample] 随机抽样 {len(sample_df)} 条文本块：")
    for idx, row in sample_df.iterrows():
        print("\n--- 随机样本 ---")
        print(f"doc_id      : {row.get('doc_id')}")
        print(f"chunk_id    : {row.get('chunk_id')}")
        print(f"chunk_index : {row.get('chunk_index')}")
        print(f"total_chunks: {row.get('total_chunks')}")
        print(f"token_count : {row.get('token_count')}")
        text = str(row.get("text", ""))
        preview = text[:300] + ("..." if len(text) > 300 else "")
        print(f"text preview: {preview}")

    print("\n[done] 文本块质量检查（基础版）完成。")


if __name__ == "__main__":
    main()
