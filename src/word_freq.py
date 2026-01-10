"""
word_freq.py

功能：
- 对 PubMed RCT train 集的 text 字段做简单的英文词频统计
- 观察高频 token，作为“领域语言特性”的数据支撑
"""

from datasets import load_dataset
from collections import Counter
import re


def tokenize(text: str):
    """
    简单英文分词：
    - 全部转小写
    - 用正则把非字母/数字的字符替换成空格
    - 再按空格切分
    说明：
    - 这是一个非常粗糙的 tokenizer，只用于做大致的高频词分析，
      不追求精准的 NLP 效果。
    """
    text = text.lower().strip()
    # 非 a-z0-9 的字符都当作分隔符
    text = re.sub(r"[^a-z0-9]+", " ", text)
    if not text:
        return []
    return text.split()


def main():
    print("[word_freq] 正在加载数据集 armanc/pubmed-rct20k ...")
    ds = load_dataset("armanc/pubmed-rct20k")
    train = ds["train"]

    counter = Counter()

    print("[word_freq] 正在遍历 train split 并统计词频（可能需要一点时间）...")
    for row in train:
        text = str(row.get("text", ""))
        tokens = tokenize(text)
        if not tokens:
            continue
        counter.update(tokens)

    print("\n[word_freq] 频率最高的前 50 个 token：")
    for word, freq in counter.most_common(50):
        print(f"{word:15s} {freq}")


if __name__ == "__main__":
    main()
