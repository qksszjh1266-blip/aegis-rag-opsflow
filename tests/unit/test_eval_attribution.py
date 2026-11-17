"""真实粗召回与最终上下文决定瀑布分支，不能以重新检索的结果替代。"""

import pytest

from aegis.db.models import BadcaseCategory
from aegis.eval.attribution import attribute_failure
from aegis.rag.retriever import RetrievalCandidate

TARGET = RetrievalCandidate("p-target", "target", "净重 0.5kg", "规格")
OTHER = RetrievalCandidate("p-other", "other", "其他文档", "规格")
WRONG_PARENT = RetrievalCandidate("p-other", "target", "售后规则", "规则")


@pytest.mark.parametrize(
    "passed,raw,final,expected",
    [
        (True, [], [], BadcaseCategory.NONE),
        (False, [OTHER], [OTHER], BadcaseCategory.MISSED_RECALL),
        (False, [OTHER, TARGET], [OTHER], BadcaseCategory.LOW_RANKING),
        (False, [OTHER, TARGET], [TARGET], BadcaseCategory.HALLUCINATION),
    ],
)
def test_waterfall_all_branches(passed, raw, final, expected):
    assert attribute_failure(passed, raw, final, ["target"]) == expected


def test_document_hit_cannot_hide_wrong_parent_or_missing_feature():
    assert (
        attribute_failure(
            False, [WRONG_PARENT], [WRONG_PARENT], ["target"], expected_parent_ids=["p-target"]
        )
        == BadcaseCategory.MISSED_RECALL
    )
    assert (
        attribute_failure(
            False,
            [WRONG_PARENT, TARGET],
            [WRONG_PARENT],
            ["target"],
            expected_header_keywords=["净重"],
        )
        == BadcaseCategory.LOW_RANKING
    )
    # 同标题不能令另一个文档越过文档编码约束。
    assert (
        attribute_failure(False, [OTHER], [OTHER], ["target"], expected_header_keywords=["规格"])
        == BadcaseCategory.MISSED_RECALL
    )
