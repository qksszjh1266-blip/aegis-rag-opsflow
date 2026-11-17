"""严格按真实粗召回与生成上下文执行三级瀑布归因。"""

from collections.abc import Collection, Sequence
from typing import Any

from aegis.db.models import BadcaseCategory
from aegis.eval.metrics import matches_target


def attribute_failure(
    passed: bool,
    raw_candidates: Sequence[Any],
    final_hits: Sequence[Any],
    expected_doc_codes: Collection[str],
    *,
    expected_parent_ids: Collection[str] = (),
    expected_header_keywords: Collection[str] = (),
) -> BadcaseCategory:
    """通过优先返回 NONE；失败时依次判定召回缺失、排序落后、上下文内答错。

    这里不推测任何缺失的轨迹，也不重新检索；调用方必须传入业务问答的同次记录。
    多目标的部分覆盖交由 Recall 指标呈现，归因以任一相关目标是否进入窗口判定。
    """
    if passed:
        return BadcaseCategory.NONE

    def relevant(hit: Any) -> bool:
        return matches_target(
            hit,
            expected_doc_codes,
            expected_parent_ids=expected_parent_ids,
            expected_header_keywords=expected_header_keywords,
        )

    if not any(relevant(hit) for hit in raw_candidates):
        return BadcaseCategory.MISSED_RECALL
    if not any(relevant(hit) for hit in final_hits):
        return BadcaseCategory.LOW_RANKING
    return BadcaseCategory.HALLUCINATION
