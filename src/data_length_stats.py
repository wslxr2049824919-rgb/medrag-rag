"""
data_length_stats.py

功能：
- 对 PubMed RCT 数据集（armanc/pubmed-rct20k）的 text 字段做长度分布分析
- 统计字符数、单词数的分布（均值、中位数、p90、p95、p99等）
- 粗略估算在 512 token 上限下，可能需要切分的“长尾”比例
  （这里用字符数做一个保守估计，例如 512 token ~ 2000 字符量级）
"""

from datasets import load_dataset
import pandas as pd


def main():
    print("[data_length_stats] 正在加载数据集 armanc/pubmed-rct20k ...")
    ds = load_dataset("armanc/pubmed-rct20k")

    train = ds["train"]

    print("[data_length_stats] 将 train split 转为 pandas DataFrame ...")
    df = train.to_pandas()

    # 1) 计算字符长度和“简单单词数”
    #    - 字符长度：len(text)
    #    - 单词数：按空格 split，粗略统计（对英文足够）
    text_series = df["text"].astype(str)

    df["char_len"] = text_series.str.len()
    df["word_len"] = text_series.str.split().str.len()

    print("\n[data_length_stats] 样本量（train）：", len(df))

    # 2) 打印字符长度的描述统计
    char_desc = df["char_len"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    print("\n[data_length_stats] 字符长度分布（char_len）：")
    print(char_desc)

    # 3) 打印单词数的描述统计
    word_desc = df["word_len"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    print("\n[data_length_stats] 单词数分布（word_len）：")
    print(word_desc)

    # 4) 粗略估算：如果嵌入模型上限约为 512 token
    #    对英文而言，512 token 大致对应 1500~2500 字符量级
    #    这里取一个保守阈值，例如 2000 字符，看看超过的比例
    approx_char_limit_for_512_tokens = 2000

    long_mask = df["char_len"] > approx_char_limit_for_512_tokens
    long_ratio = long_mask.mean()
    long_count = long_mask.sum()

    print(
        f"\n[data_length_stats] 估算：以 ~512 token 上限，"
        f"取字符阈值 {approx_char_limit_for_512_tokens}："
    )
    print(f"  - 超过该长度的样本数：{long_count}")
    print(f"  - 占比约：{long_ratio:.4%}")

    # 5) 看看最长的几条 text 长什么样，帮助理解“长尾”情况
    print("\n[data_length_stats] 字符长度最长的前 5 条样本：")
    top5 = df.sort_values("char_len", ascending=False).head(5)
    for i, row in top5.iterrows():
        print("\n--- 样本 ---")
        print(f"char_len = {row['char_len']}, word_len = {row['word_len']}")
        # 只打印前 400 字符，避免刷屏
        txt = str(row["text"])
        print(txt[:400] + ("..." if len(txt) > 400 else ""))


if __name__ == "__main__":
    main()
