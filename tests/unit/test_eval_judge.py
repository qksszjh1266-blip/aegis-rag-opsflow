"""百分比容差、量纲等价、语义裁判协议与 Token 双账本隔离测试。"""

import json
from decimal import Decimal

import httpx
import pytest

from aegis.config import Settings
from aegis.eval.judge import (
    AnswerJudge,
    AppTokenLedger,
    JudgeTokenLedger,
    NumericJudge,
)


@pytest.mark.parametrize(
    "answer,reference,passed",
    [
        ("0.0523", "5.2%", True),
        ("0.085", "5.2%", False),
        ("5.35%", "5.2%", True),
        ("5.3501%", "5.2%", False),
        ("百分之5.23", "5.2%", True),
        ("0.5kg", "500g", True),
        ("1.5L", "1500ml", True),
        ("额定功率2.1 kW", "额定功率2100 W", True),
        ("500kg", "500g", False),
        ("500g", "500ml", False),
        ("5元", "5%", False),
        ("5.2", "5.2%", False),
        ("净重不是500g", "净重500g", False),
        ("赔付5.2%，另赔付8.5%", "赔付5.2%", False),
        ("重量500g，容量1.5L", "重量500g，容量1500ml", True),
        ("35–98℃", "35–98 ℃", True),
        ("98℃", "35–98 ℃", False),
        ("1℃", "±1 ℃", False),
        ("0.001kg", "0g", False),
        ("0g", "0g", True),
        ("1万元", "10000元", True),
        ("60分钟", "1小时", True),
        ("30天", "1个月", False),
        ("1天", "1个工作日", False),
        ("3件", "3台", False),
        ("结果约为1.0005", "结果为1", True),
        ("结果约为1.01", "结果为1", False),
    ],
)
def test_numeric_percentage_and_units(answer, reference, passed):
    assert NumericJudge().compare(answer, reference).passed is passed


def test_equal_representations_merge_but_distinct_values_require_separate_claims():
    judge = NumericJudge()
    result = judge.compare(
        "净重500g，容量1.5L", "净重0.5kg，容量1500ml", ["0.5kg", "500g", "1500ml", "1.5L"]
    )
    assert result.passed and result.expected_count == result.matched_count == 2
    assert not judge.compare("5.2%", "5.2%与5.3%", ["5.2%", "5.3%"]).passed
    assert judge.compare("5.2%与5.3%", "5.2%与5.3%", ["5.2%", "5.3%"]).passed


def test_expected_numbers_allow_reference_side_details_but_not_unexpected_errors():
    judge = NumericJudge()
    reference = "按5%赔付，最低5元，最高50元。"
    assert judge.compare("按5%赔付，最高50元。", reference, ["5%"]).passed
    assert not judge.compare("按5%赔付，最高100元。", reference, ["5%"]).passed


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1")])
def test_invalid_tolerance_rejected(bad):
    with pytest.raises(ValueError):
        NumericJudge(percentage_point_tolerance=bad)


async def test_mock_judge_records_estimated_tokens_and_rejects_refusal():
    judge = AnswerJudge(Settings(_env_file=None))
    result = await judge.judge(
        query="净重多少",
        answer="净重500g",
        reference_answer="净重0.5kg",
        expected_numbers=["0.5kg"],
    )
    assert result.passed and result.numeric_passed
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    assert (
        judge.ledger.snapshot()["total_tokens"] == result.prompt_tokens + result.completion_tokens
    )
    failed = await judge.judge(
        query="参数", answer="证据不足，无法确认", reference_answer="标准规格", expected_numbers=[]
    )
    assert not failed.passed
    equivalent = await judge.judge(query="比例", answer="0.0523", reference_answer="5.2%")
    assert equivalent.passed
    await judge.aclose()


def test_app_and_judge_ledgers_are_separate_instances_and_types():
    app = AppTokenLedger()
    judge = JudgeTokenLedger()
    app.record(10, 20)
    judge.record(30, 40)
    assert app.snapshot() == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "calls": 1,
    }
    assert judge.snapshot() == {
        "prompt_tokens": 30,
        "completion_tokens": 40,
        "total_tokens": 70,
        "calls": 1,
    }
    with pytest.raises(TypeError):
        AnswerJudge(Settings(_env_file=None), ledger=app)
    with pytest.raises(ValueError):
        judge.record(True, 2)
    assert judge.snapshot()["total_tokens"] == 70


def remote_settings():
    return Settings(
        _env_file=None,
        provider_mode="dashscope",
        llm_api_key="test-key",
        llm_base_url="https://judge.example/v1",
    )


@pytest.mark.parametrize(
    "content,finish,passed,error",
    [
        (
            json.dumps({"score": 0.95, "passed": True, "reason": "对象与语义一致"}),
            "stop",
            True,
            None,
        ),
        (json.dumps({"score": 0.5, "passed": True, "reason": "覆盖不足"}), "stop", False, None),
        ("损坏的JSON", "stop", False, "ValidationError"),
        (
            json.dumps({"score": 0.95, "passed": True, "reason": "正确"}),
            "length",
            False,
            "ValueError",
        ),
        (
            json.dumps({"score": 1.2, "passed": True, "reason": "正确"}),
            "stop",
            False,
            "ValidationError",
        ),
    ],
)
async def test_remote_judge_records_usage_before_validating_verdict(content, finish, passed, error):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 41, "completion_tokens": 17},
                "choices": [{"finish_reason": finish, "message": {"content": content}}],
            },
        )

    ledger = JudgeTokenLedger()
    app_ledger = AppTokenLedger()
    app_ledger.record(100, 20)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = AnswerJudge(remote_settings(), ledger=ledger, http_client=client)
        result = await judge.judge(query="净重？", answer="净重500g", reference_answer="净重0.5kg")
        assert result.passed is passed and result.error == error
        assert result.prompt_tokens == 41 and result.completion_tokens == 17
        await judge.aclose()
        assert not client.is_closed
    assert requests[0]["enable_thinking"] is False
    assert "max_tokens" in requests[0]
    assert ledger.snapshot()["total_tokens"] == 58
    assert app_ledger.snapshot()["total_tokens"] == 120


async def test_numeric_failure_does_not_call_remote_semantic_judge():
    def respond(request):
        pytest.fail("确定性失败不能交给语义裁判翻案")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = AnswerJudge(remote_settings(), http_client=client)
        result = await judge.judge(query="比例", answer="0.085", reference_answer="5.2%")
    assert not result.passed and not result.numeric_passed
    assert judge.ledger.snapshot()["calls"] == 0


async def test_judge_failure_is_explicit_and_does_not_invent_usage():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        judge = AnswerJudge(remote_settings(), http_client=client)
        result = await judge.judge(query="比例", answer="5.2%", reference_answer="5.2%")
    assert not result.passed and result.error == "HTTPStatusError"
    assert result.prompt_tokens == result.completion_tokens == 0


@pytest.mark.parametrize("answer", ["5.2%是错误的", "错误答案为5.2%", "不适用5.2%"])
def test_negated_and_rejected_claims_cannot_satisfy_numeric_reference(answer):
    assert not NumericJudge().compare(answer, "5.2%").passed


def test_percentage_equivalent_annotations_do_not_depend_on_order():
    for expected in (["0.052", "5.2%"], ["5.2%", "0.052"]):
        result = NumericJudge().compare("0.0523", "比例为0.052，即5.2%", expected)
        assert result.passed and result.expected_count == 1


@pytest.mark.parametrize(
    "answer,reference",
    [
        (
            "SKU-A 净重0.8kg，SKU-B 净重0.5kg",
            "SKU-A 净重0.5kg，SKU-B 净重0.8kg",
        ),
        (
            "SKU-A 净重0.8kg，毛重0.5kg",
            "SKU-A 净重0.5kg，毛重0.8kg",
        ),
        (
            "SKU-A 额定功率2000W，最大功率1200W",
            "SKU-A 额定功率1200W，最大功率2000W",
        ),
        (
            "净重0.5kg、净重0.8kg",
            "SKU-A 净重0.5kg，SKU-B 净重0.8kg",
        ),
        (
            "SKU-A 重量0.5kg、0.8kg",
            "SKU-A 净重0.5kg，毛重0.8kg",
        ),
        ("SKU-B 净重500g", "SKU-A 净重500g"),
    ],
)
async def test_swapped_or_ambiguous_numeric_bindings_are_rejected(answer, reference):
    numeric = NumericJudge().compare(answer, reference)
    assert not numeric.passed
    assert any("参数归属" in item for item in numeric.details)
    judge = AnswerJudge(Settings(_env_file=None))
    result = await judge.judge(query="核对参数", answer=answer, reference_answer=reference)
    assert not result.passed and not result.numeric_passed


@pytest.mark.parametrize(
    "answer,reference",
    [
        (
            "SKU-B 净重800g；SKU-A 净重500g",
            "SKU-A 净重0.5kg，SKU-B 净重0.8kg",
        ),
        (
            "SKU-A 包装毛重800g，净质量500g",
            "SKU-A 净重0.5kg，毛重0.8kg",
        ),
        ("净重500g", "SKU-A 净重0.5kg"),
        ("500g", "SKU-A 净重0.5kg"),
    ],
)
def test_consistent_bindings_allow_reordering_aliases_and_unambiguous_ellipsis(answer, reference):
    assert NumericJudge().compare(answer, reference).passed


async def test_binding_failure_prevents_real_judge_from_overriding_wrong_assignment():
    def respond(request):
        pytest.fail("参数归属错误不能交给真实裁判翻案")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        judge = AnswerJudge(remote_settings(), http_client=client)
        result = await judge.judge(
            query="净重和毛重多少",
            answer="SKU-A 净重0.8kg，毛重0.5kg",
            reference_answer="SKU-A 净重0.5kg，毛重0.8kg",
        )
    assert not result.passed and judge.ledger.snapshot()["calls"] == 0


@pytest.mark.parametrize(
    "answer,reference",
    [
        ("SKU 编码：DEV-mini；净重：850 g。", "DEV-mini 的净重为0.85kg，即850g。"),
        ("SKU 编码：DEV-std；容量：1.5 L。", "DEV-std 的容量为1500ml，即1.5L。"),
        ("按0.0523赔付。", "按5.2%赔付。"),
    ],
)
async def test_equivalent_numeric_wording_does_not_fail_semantic_scoring(answer, reference):
    judge = AnswerJudge(Settings(_env_file=None))
    result = await judge.judge(query="核对参数", answer=answer, reference_answer=reference)
    assert result.passed and result.numeric_passed
    assert result.prompt_tokens > 0 and result.completion_tokens > 0
    assert (
        judge.ledger.snapshot()["total_tokens"] == result.prompt_tokens + result.completion_tokens
    )


async def test_mock_semantic_failure_is_counted_but_numeric_failure_is_not():
    judge = AnswerJudge(Settings(_env_file=None))
    first = await judge.judge(
        query="库位校验可以跳过吗", answer="可以跳过库位校验", reference_answer="禁止跳过库位校验"
    )
    assert not first.passed and first.prompt_tokens > 0
    snapshot = judge.ledger.snapshot()
    numeric_failure = await judge.judge(query="比例", answer="0.085", reference_answer="5.2%")
    assert not numeric_failure.passed
    assert numeric_failure.prompt_tokens == numeric_failure.completion_tokens == 0
    assert judge.ledger.snapshot() == snapshot


def test_evidence_can_support_extra_fields_without_replacing_required_truth():
    evidence = [
        "| SKU | 净重 | 容量 |\n| --- | --- | --- |\n"
        "| DEV-A | 0.5kg | 1.5L |\n| DEV-B | 0.8kg | 2L |\n"
    ]
    judge = NumericJudge()
    reference = "DEV-A 净重0.5kg"
    assert judge.compare("DEV-A 净重500g，容量1.5L", reference, evidence=evidence).passed
    assert not judge.compare("DEV-A 净重500g，容量2L", reference, evidence=evidence).passed
    assert not judge.compare("DEV-A 容量1.5L", reference, evidence=evidence).passed
    assert not judge.compare("DEV-A 净重500g，净重800g", reference, evidence=evidence).passed


async def test_mock_tokens_remain_physically_separate_from_application_ledger():
    app_ledger = AppTokenLedger()
    app_ledger.record(7, 3)
    judge = AnswerJudge(Settings(_env_file=None))
    result = await judge.judge(query="重量", answer="净重500g", reference_answer="净重0.5kg")
    assert result.passed and judge.ledger.snapshot()["total_tokens"] > 0
    assert app_ledger.snapshot()["total_tokens"] == 10


@pytest.mark.parametrize(
    "answer,reference",
    [
        (
            "配件编码FLT-02；适配型号DEV-Pro；更换周期6个月。",
            "DEV-Pro 适配FLT-02滤芯，更换周期6个月。",
        ),
        (
            "批量异常转P0_EMERGENCY工单；单点处置超过10分钟仍失败建工单。",
            "单点处置超过10分钟仍失败建工单。",
        ),
        (
            "扫描库位HZ-R01；质检安全异常转P0_EMERGENCY；批量拦截不少于10件建P1_HIGH。",
            "扫描库位HZ-R01；批量拦截不少于10件建P1_HIGH。",
        ),
        (
            "DEV-B系列说明。质量问题不受7天限制，签收15天内换修，整机保修1年。",
            "质量问题不受7天限制，签收15天内换修，整机保修1年。",
        ),
    ],
)
def test_policy_quantities_do_not_inherit_unrelated_sku_or_ticket_identifiers(answer, reference):
    assert NumericJudge().compare(answer, reference).passed


def test_policy_extra_numbers_need_actual_evidence_after_identifier_boundary():
    reference = "DEV-A 系列整机保修1年。"
    answer = "DEV-A 系列整机保修1年。DEV-B 系列电机保修3年。"
    assert not NumericJudge().compare(answer, reference).passed
    assert NumericJudge().compare(answer, reference, evidence=[answer]).passed
    assert (
        not NumericJudge()
        .compare(answer.replace("3年", "4年"), reference, evidence=[answer])
        .passed
    )


def test_policy_lower_and_upper_limits_cannot_be_swapped():
    judge = NumericJudge()
    reference = "按5%赔付，最低5元、最高50元。"
    assert not judge.compare("按5%赔付，最低50元、最高5元。", reference).passed
    assert judge.compare("按5%赔付，下限5元、单笔封顶50元。", reference).passed
    assert judge.compare("按5%赔付，最低5元、最高50元。", reference).passed


def test_unlabelled_final_payment_does_not_inherit_previous_sentence_limit():
    reference = "实付2000元按5%计算100元，按上限赔付50元。"
    assert (
        NumericJudge().compare("按5%计算100元，最终赔付50元。", reference, ["100元", "50元"]).passed
    )
