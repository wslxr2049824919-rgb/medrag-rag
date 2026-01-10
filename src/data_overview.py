"""
data_overview.py

功能：
- 对 PubMed RCT 数据集（armanc/pubmed-rct20k）做基础结构和缺失情况分析
- 输出每个数据划分的大小和字段
- 统计各字段的 NaN 缺失比例
- 专门检查 text 字段：空字符串、极短文本的比例和示例
"""

from datasets import load_dataset
import pandas as pd


def main():
    print("[data_overview] 正在加载数据集 armanc/pubmed-rct20k ...")
    ds = load_dataset("armanc/pubmed-rct20k")

    # 1) 打印各个 split 的大小和字段名
    print("\n[data_overview] 各数据集划分（split）的行数和字段：")
    for split_name, split in ds.items():
        print(f"  - {split_name}: {len(split)} 行, 字段 = {split.column_names}")

    # 后续分析先聚焦在 train 集
    train = ds["train"]

    # 2) 转成 pandas DataFrame，便于做统计
    print("\n[data_overview] 正在将 train split 转为 pandas DataFrame ...")
    df = train.to_pandas()

    print("\n[data_overview] DataFrame 形状（train）：", df.shape)

    # 3) 每一列的 NaN 缺失比例
    print("\n[data_overview] 各字段 NaN 缺失比例：")
    na_ratio = df.isna().mean()
    print(na_ratio)

    # 4) 对 text 字段做更细一点的质量分析
    if "text" in df.columns:
        # 将 text 转成字符串，strip 掉首尾空白
        text_series = df["text"].astype(str)

        # 空字符串或只有空白字符
        empty_mask = text_series.str.strip().eq("")
        empty_ratio = empty_mask.mean()

        # “极短文本”示例：长度 < 10 个字符（你可以根据需要调整这个阈值）
        length_series = text_series.str.len()
        short_mask = length_series < 10
        short_ratio = short_mask.mean()

        print("\n[data_overview] `text` 字段质量检查：")
        print(f"  - 空字符串或仅空白：{empty_ratio:.4%}")
        print(f"  - 极短文本（长度 < 10 字符）：{short_ratio:.4%}")

        # 打印几条极短文本的样本，看看是不是“垃圾数据”
        print("\n[data_overview] 极短 `text` 示例（最多 10 条）：")
        short_examples = text_series[short_mask].head(10)
        for i, val in enumerate(short_examples, 1):
            print(f"  [{i}] {repr(val)}")

    else:
        print("\n[data_overview] train split 中没有 `text` 字段，需检查数据集结构。")


if __name__ == "__main__":
    main()

