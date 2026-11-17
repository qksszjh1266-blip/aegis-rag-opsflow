"""检索质量与延迟纯函数：固定截断窗口，重复命中不会重复计分。"""

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Literal


def _validate_k(k: int) -> None:
    if type(k) is not int or k < 1:
        raise ValueError("K 必须为正整数")


def hit_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int = 5) -> float:
    """前 K 个位置至少出现一个相关目标记 1，否则记 0；空目标记 0。"""
    _validate_k(k)
    return float(bool(set(retrieved[:k]) & set(relevant)))


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int = 5) -> float:
    """分母为完整目标集合而非召回池；重复目标和重复结果均只计算一次。"""
    _validate_k(k)
    expected = set(relevant)
    return len(set(retrieved[:k]) & expected) / len(expected) if expected else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Collection[str], k: int | None = None
) -> float:
    """首个相关目标的原始排名倒数；未命中或无相关目标时为零。"""
    if k is not None:
        _validate_k(k)
    expected = set(relevant)
    for rank, identifier in enumerate(retrieved if k is None else retrieved[:k], start=1):
        if identifier in expected:
            return 1 / rank
    return 0.0


def mrr(
    rankings: Sequence[Sequence[str]],
    relevant_sets: Sequence[Collection[str]],
    k: int | None = None,
) -> float:
    """逐样本倒数排名取算术平均，未命中样本保留在分母内。"""
    if len(rankings) != len(relevant_sets):
        raise ValueError("检索排名与目标集合的样本数量必须一致")
    if k is not None:
        _validate_k(k)
    scores = [
        reciprocal_rank(items, targets, k)
        for items, targets in zip(rankings, relevant_sets, strict=True)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    """使用线性插值分位数；空样本返回零，拒绝非有限数与非法百分位。"""
    if isinstance(q, bool) or not math.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("百分位必须是 0 至 100 的有限数值")
    ordered = sorted(values)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in ordered):
        raise ValueError("分位数样本必须为有限数值")
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def latency_percentiles(latencies_ms: Iterable[float]) -> dict[str, float]:
    """延迟统一以毫秒统计；空运行的两个分位数均为零。"""
    values = list(latencies_ms)
    if any(value < 0 for value in values):
        raise ValueError("端到端延迟不能为负数")
    return {"p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95)}


def _value(hit: Any, name: str) -> str:
    value = hit.get(name, "") if isinstance(hit, Mapping) else getattr(hit, name, "")
    return value if isinstance(value, str) else ""


def matches_target(
    hit: Any,
    expected_doc_codes: Collection[str],
    *,
    expected_parent_ids: Collection[str] = (),
    expected_header_keywords: Collection[str] = (),
) -> bool:
    """文档限制始终生效；父块 ID 优先，否则标题或正文命中任一特征即可。"""
    if _value(hit, "doc_code") not in expected_doc_codes:
        return False
    if expected_parent_ids:
        return _value(hit, "parent_id") in expected_parent_ids
    if expected_header_keywords:
        text = _value(hit, "header_path") + "\n" + _value(hit, "content_text")
        return any(keyword and keyword in text for keyword in expected_header_keywords)
    return True


def retrieval_metrics(
    hits: Sequence[Any],
    expected_doc_codes: Collection[str],
    *,
    expected_parent_ids: Collection[str] = (),
    expected_header_keywords: Collection[str] = (),
    k: int = 5,
    granularity: Literal["doc_code", "parent_id"] = "doc_code",
) -> dict[str, float]:
    """双粒度指标；父块 ID 应由完整基准或知识库解析，不允许仅从召回池构建。

    未提供父块 ID 时，每个非空特征构成一个完整目标；Recall 为特征覆盖率，
    Hit 与倒数排名仍按父块原始位置统计。文档粒度也保留原始父块窗口和位置。
    """
    _validate_k(k)
    window = hits[:k]
    if granularity == "doc_code":
        ranked = [_value(hit, "doc_code") for hit in window]
        return {
            "hit_at_k": hit_at_k(ranked, expected_doc_codes, k),
            "recall_at_k": recall_at_k(ranked, expected_doc_codes, k),
            "mrr": reciprocal_rank(ranked, expected_doc_codes, k),
        }
    if granularity != "parent_id":
        raise ValueError("检索粒度必须为 doc_code 或 parent_id")
    if expected_parent_ids:
        ranked = [
            _value(hit, "parent_id") if _value(hit, "doc_code") in expected_doc_codes else ""
            for hit in window
        ]
        return {
            "hit_at_k": hit_at_k(ranked, expected_parent_ids, k),
            "recall_at_k": recall_at_k(ranked, expected_parent_ids, k),
            "mrr": reciprocal_rank(ranked, expected_parent_ids, k),
        }
    targets = {keyword for keyword in expected_header_keywords if keyword}
    if not targets:
        return {"hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    covered: set[str] = set()
    first_rank: int | None = None
    for rank, hit in enumerate(window, start=1):
        if _value(hit, "doc_code") not in expected_doc_codes:
            continue
        text = _value(hit, "header_path") + "\n" + _value(hit, "content_text")
        matched = {keyword for keyword in targets if keyword in text}
        covered.update(matched)
        if matched and first_rank is None:
            first_rank = rank
    return {
        "hit_at_k": float(bool(covered)),
        "recall_at_k": len(covered) / len(targets),
        "mrr": 1 / first_rank if first_rank is not None else 0.0,
    }
