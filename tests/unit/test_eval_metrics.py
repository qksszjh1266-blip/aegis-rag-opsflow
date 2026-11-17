"""固定 Top-K 窗口、双粒度召回与分位数边界的确定性验证。"""

import pytest

from aegis.eval.metrics import (
    hit_at_k,
    latency_percentiles,
    matches_target,
    mrr,
    percentile,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)


@pytest.mark.parametrize(
    "ranked,expected,k,hit,recall,rr",
    [
        ([], ["a"], 5, 0, 0, 0),
        (["a"], [], 5, 0, 0, 0),
        (["a", "a", "b"], ["a", "b", "b"], 2, 1, 0.5, 1),
        (["x", "a", "b"], ["a", "b"], 1, 0, 0, 0),
        (["x", "a", "b"], ["a", "b"], 2, 1, 0.5, 0.5),
        (["x", "a", "b"], ["a", "b"], 9, 1, 1, 0.5),
    ],
)
def test_hit_recall_reciprocal_rank_boundaries(ranked, expected, k, hit, recall, rr):
    assert hit_at_k(ranked, expected, k) == hit
    assert recall_at_k(ranked, expected, k) == recall
    assert reciprocal_rank(ranked, expected, k) == rr


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_window_rejected(k):
    for function in (hit_at_k, recall_at_k, reciprocal_rank):
        with pytest.raises(ValueError):
            function(["a"], ["a"], k)


def test_mrr_includes_misses_and_rejects_misaligned_samples():
    assert mrr([["x", "a"], ["x"], ["a"]], [{"a"}, {"a"}, {"a"}]) == 0.5
    assert mrr([], []) == 0
    assert mrr([["x", "a"]], [{"a"}], k=1) == 0
    with pytest.raises(ValueError):
        mrr([["a"]], [])


def test_linear_latency_percentiles_and_empty_sample():
    assert latency_percentiles([40, 10, 30, 20]) == {"p50_ms": 25, "p95_ms": 38.5}
    assert latency_percentiles([]) == {"p50_ms": 0, "p95_ms": 0}
    assert percentile([9], 95) == 9
    assert percentile([10, 20], 0) == 10
    assert percentile([10, 20], 100) == 20
    with pytest.raises(ValueError):
        latency_percentiles([-1])
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            percentile([value], 95)
    for q in (-1, 101, float("nan"), True):
        with pytest.raises(ValueError):
            percentile([1], q)


def test_document_window_not_compressed_by_duplicate_parent_document():
    hits = [
        {"doc_code": "x", "parent_id": "x1"},
        {"doc_code": "x", "parent_id": "x2"},
        {"doc_code": "target", "parent_id": "p1"},
    ]
    assert retrieval_metrics(hits, ["target"], k=2) == {"hit_at_k": 0, "recall_at_k": 0, "mrr": 0}
    assert retrieval_metrics(hits, ["target"], k=3)["mrr"] == pytest.approx(1 / 3)


def test_parent_ids_and_feature_targets_keep_full_expected_denominator():
    hits = [
        {"doc_code": "other", "parent_id": "p1", "header_path": "容量"},
        {"doc_code": "target", "parent_id": "p1", "header_path": "规格", "content_text": "净重"},
    ]
    expected = {"hit_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5}
    assert (
        retrieval_metrics(
            hits, ["target"], expected_parent_ids=["p1", "p2"], granularity="parent_id"
        )
        == expected
    )
    assert (
        retrieval_metrics(
            hits, ["target"], expected_header_keywords=["净重", "容量"], granularity="parent_id"
        )
        == expected
    )
    assert not matches_target(hits[0], ["target"], expected_parent_ids=["p1"])
    assert matches_target(hits[1], ["target"], expected_header_keywords=["净重", "容量"])
    assert not matches_target(
        hits[1], ["target"], expected_parent_ids=["p2"], expected_header_keywords=["净重"]
    )
    assert retrieval_metrics([], ["target"], granularity="parent_id")["recall_at_k"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics(hits, ["target"], granularity="未知")
