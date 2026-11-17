"""稳定分词和 BM25 稀疏编码；IDF 由 Qdrant 按当前集合统计计算。"""

import hashlib
import re
from collections import Counter

from qdrant_client import models

# 固定长度先验让增量文档无需重算全库 TF；修改参数必须完整重建集合。
SPARSE_VERSION = "bm25-cjk-v1-k1=1.2-b=0.75-avgdl=200"
_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u3400-\u9fff]|℃", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """英文/SKU 按词、中文按单字及相邻双字编码，兼顾编号与中文短语。"""
    tokens = _TOKEN.findall(text.lower())
    bigrams = [
        text[i : i + 2]
        for i in range(len(text) - 1)
        if all("\u3400" <= char <= "\u9fff" for char in text[i : i + 2])
    ]
    return tokens + bigrams


def _counts(text: str) -> Counter[int]:
    # 使用固定哈希而非 Python hash，避免进程重启后词项编号发生变化。
    return Counter(
        int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
        for token in tokenize(text)
    )


def encode_document(text: str) -> models.SparseVector:
    """BM25 的 TF 饱和与长度归一项；稀疏 IDF modifier 在查询侧补齐 IDF。"""
    counts = _counts(text)
    length = sum(counts.values())
    indices = sorted(counts)
    return models.SparseVector(
        indices=indices,
        values=[
            counts[i] * 2.2 / (counts[i] + 1.2 * (0.25 + 0.75 * length / 200)) for i in indices
        ],
    )


def encode_query(text: str) -> models.SparseVector:
    """查询词去重，避免重复输入人为放大得分。"""
    indices = sorted(_counts(text))
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))
