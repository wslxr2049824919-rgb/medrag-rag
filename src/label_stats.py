"""
label_stats.py

功能：
- 统计 PubMed RCT 数据集（armanc/pubmed-rct20k）中 label 字段的取值分布
- 用于回答：
  1）label 实际有哪些取值？
  2）每种 label 各有多少条样本？
"""

from datasets import load_dataset
import pandas as pd


def main():
    print("[label_stats] 正在加载数据集 armanc/pubmed-rct20k ...")
    ds = load_dataset("armanc/pubmed-rct20k")
    train = ds["train"]

    print("[label_stats] 正在将 train split 转为 pandas DataFrame ...")
    df = train.to_pandas()

    # 1) label 的频数统计（每个标签有多少条）
    print("\n[label_stats] label 字段取值频数统计（按数量从大到小）：")
    value_counts = df["label"].value_counts()
    print(value_counts)

    # 2) label 的去重取值列表（方便在文档里写完整的标签集合）
    print("\n[label_stats] label 字段去重后的取值列表：")
    unique_labels = sorted(df["label"].unique().tolist())
    print(unique_labels)


if __name__ == "__main__":
    main()

