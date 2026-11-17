"""前置零调用匹配与 SKU 逐行量纲核验的确定性边界。"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from aegis.guardrails.precheck import HighRiskPrecheck
from aegis.guardrails.sku_verifier import SKUVerifier, extract_quantities, load_sku_rows


@pytest.fixture
def rows():
    return [
        {
            "SKU": "HEAT-01",
            "净重": "0.5 kg",
            "毛重": "0.8 kg",
            "容量": "1.5 L",
            "额定功率": "1.2 kW",
            "工作温度": "-10–95 ℃",
            "温控精度": "±1 ℃",
        },
        {
            "SKU": "HEAT-02",
            "净重": "0.8 kg",
            "毛重": "1.1 kg",
            "容量": "2 L",
            "额定功率": "1.5 kW",
            "工作温度": "5–100 ℃",
            "温控精度": "±2 ℃",
        },
    ]


@pytest.mark.parametrize("word", ["漏电", "冒烟", "起火", "批量资损", "监管投诉"])
def test_all_mandatory_redlines_remain_even_with_custom_configuration(word):
    check = HighRiskPrecheck(["破损升级"])
    assert word in check.check(f"发现{word}，请处理")
    assert check.check("包装正常") == ()
    assert "破损升级" in check.check("申请破损升级")


def test_precheck_handles_zero_width_and_separated_characters():
    assert HighRiskPrecheck().check("漏\u200b电和监管 投诉") == ("漏电", "监管投诉")
    with pytest.raises(ValueError):
        HighRiskPrecheck("冒烟")
    with pytest.raises(ValueError):
        HighRiskPrecheck([""])


@pytest.mark.parametrize(
    "answer",
    [
        "HEAT-01 净重500g，容量1500ml，额定功率1200W。",
        "HEAT-01 净重0.5kg，容量1.5L，额定功率1.2kW。",
        "HEAT-01 净重５００ｇ，容量１．５Ｌ。",
        "HEAT-01 净重5e2 g，额定功率1,200W。",
        "HEAT-01 工作温度 -10 ℃ 至 95 ℃，温控精度±1℃。",
        "HEAT-01 工作温度 -10–95 ℃。",
        "HEAT-01 工作温度上限95℃，工作温度下限-10℃。",
    ],
)
def test_unit_equivalence_and_exact_ranges_pass(rows, answer):
    result = SKUVerifier().verify(answer, rows, query="HEAT-01 的参数")
    assert result.passed, result.corrections
    assert result.checked_count > 0


@pytest.mark.parametrize(
    "answer",
    [
        "HEAT-01 净重800g。",
        "HEAT-01 净重500W。",
        "HEAT-01 额定功率1500W。",
        "HEAT-01 工作温度95℃。",
        "HEAT-01 温控精度±2℃。",
        "HEAT-01 净重500g/s。",
        "HEAT-01 净重500lb。",
    ],
)
def test_wrong_value_dimension_other_row_and_truncated_range_fail(rows, answer):
    result = SKUVerifier().verify(answer, rows, query="HEAT-01 的参数")
    assert not result.passed
    assert result.corrections


def test_corrupted_mass_has_specific_truth_correction(rows):
    result = SKUVerifier().verify("HEAT-01 净重800g", rows, query="HEAT-01 净重")
    assert "净重 真值为 0.5 kg" in result.corrections
    assert "800g" in result.corrections


def test_same_row_other_attribute_is_not_accepted(rows):
    result = SKUVerifier().verify("HEAT-01 净重800g，毛重500g", rows)
    assert not result.passed
    assert "净重 真值为 0.5 kg" in result.corrections
    assert "毛重 真值为 0.8 kg" in result.corrections


def test_multi_sku_comparison_is_checked_per_local_owner(rows):
    verifier = SKUVerifier()
    answer = "HEAT-01 净重500g，HEAT-02 净重800g。"
    assert verifier.verify(answer, rows, query="比较 HEAT-01 和 HEAT-02 净重").passed
    assert not verifier.verify(answer.replace("500g", "800g"), rows).passed
    assert not verifier.verify("HEAT-01 和 HEAT-02 净重都是500g。", rows).passed


def test_answer_changes_product_from_query_is_blocked(rows):
    assert not SKUVerifier().verify("HEAT-02 净重800g", rows, query="HEAT-01 净重？").passed


def test_no_identity_or_ambiguous_attribute_fails_closed(rows):
    assert not SKUVerifier().verify("净重500g", rows).passed
    assert not SKUVerifier().verify("HEAT-01 重量500g", rows).passed
    assert SKUVerifier().verify("净重500g", rows, query="HEAT-01 净重？").passed


def test_same_series_identifier_prefix_does_not_steal_variant():
    rows = [
        {"SKU": "P-01", "型号": "TH-200 Pro", "净重": "1 kg"},
        {"SKU": "P-01-L", "型号": "TH-200 Pro Lite", "净重": "0.8 kg"},
    ]
    verifier = SKUVerifier()
    assert verifier.verify("TH-200 Pro Lite 净重800g", rows).passed
    assert not verifier.verify("P-01-L 净重1kg", rows).passed


def test_conflicting_truths_for_same_sku_fail_closed():
    rows = [{"SKU": "S-01", "净重": "0.5kg"}, {"SKU": "S-01", "净重": "0.8kg"}]
    assert not SKUVerifier().verify("S-01 净重500g", rows).passed


def test_empty_evidence_allows_no_claims_but_rejects_physical_claims():
    verifier = SKUVerifier()
    assert verifier.verify("请提供商品型号，以便进一步核实。", []).passed
    assert not verifier.verify("净重500g", []).passed
    assert not verifier.verify("净重500lb", []).passed


def test_column_unit_and_legacy_structured_quantity_supported():
    verifier = SKUVerifier()
    assert verifier.verify("S-01 净重500g", [{"SKU": "S-01", "净重(kg)": "0.5"}]).passed
    rows = [{"SKU": "S-01", "重量": {"数值": 0.5, "单位": "kg"}}]
    assert verifier.verify("S-01 重量500g", rows).passed


def test_scalar_range_and_tolerance_keep_distinct_semantics():
    items = extract_quantities("-10–95℃，±1℃，1200W，0.5kg")
    assert [item.kind for item in items] == ["range", "tolerance", "scalar", "scalar"]
    assert [item.dimension for item in items] == ["temperature", "temperature", "power", "mass"]
    assert items[3].values == (500,)


async def test_sku_loader_limits_query_to_authorized_parent_ids_and_deduplicates():
    session = AsyncMock()
    row = {"SKU": "S-01", "净重": "0.5kg"}
    session.scalars.return_value = [
        {"rows": [row]},
        {"rows": [row]},
        {"S-02": {"净重": {"数值": 1, "单位": "kg"}}},
    ]

    @asynccontextmanager
    async def factory():
        yield session

    assert await load_sku_rows(factory, []) == []
    session.scalars.assert_not_awaited()
    result = await load_sku_rows(factory, ["authorized-parent"])
    assert result == [row, {"SKU": "S-02", "净重": {"数值": 1, "单位": "kg"}}]
    statement = session.scalars.call_args.args[0]
    assert list(statement.compile().params.values()) == [["authorized-parent"]]


def test_unseen_attribute_cannot_borrow_value_from_existing_one():
    rows = [{"SKU": "S-01", "净重": "0.5kg"}]
    result = SKUVerifier().verify("S-01 毛重500g", rows, query="S-01 净重？")
    assert not result.passed
    assert "没有属性" in result.corrections


def test_unknown_product_cannot_inherit_queried_product_truth(rows):
    result = SKUVerifier().verify("HEAT-99 净重500g", rows, query="HEAT-01 净重？")
    assert not result.passed
    assert "不在召回真值" in result.corrections


def test_product_identifier_with_unit_like_suffix_is_not_physical_claim():
    rows = [{"SKU": "BOX-500G", "净重": "0.5kg"}]
    assert SKUVerifier().verify("BOX-500G 净重500g", rows).passed
    assert SKUVerifier().verify("请核实 BOX-500G 的订单。", rows).passed


def test_leading_decimal_is_checked_and_missing_unit_is_rejected(rows):
    verifier = SKUVerifier()
    assert verifier.verify("HEAT-01 净重.5kg", rows).passed
    assert not verifier.verify("HEAT-01 净重.8kg", rows).passed
    assert not verifier.verify("HEAT-01 净重为800", rows).passed


@pytest.mark.parametrize(
    "query", ["SKU-Z 净重是多少？", "XYZ999 净重是多少？", "型号: UNKNOWN 净重？"]
)
def test_unknown_query_product_never_borrows_single_recalled_row(query):
    rows = [{"SKU": "SKU-A", "净重": "0.5kg"}]
    result = SKUVerifier().verify("净重500g。", rows, query=query)
    assert not result.passed
    assert "提问商品" in result.corrections


def test_temperature_upper_and_lower_bounds_have_distinct_property_identity():
    rows = [{"SKU": "SKU-A", "温度上限": "95℃", "温度下限": "5℃"}]
    verifier = SKUVerifier()
    assert verifier.verify("SKU-A 最高温度95℃，最低温度5℃。", rows).passed
    assert not verifier.verify("SKU-A 最低温度95℃。", rows).passed
    assert not verifier.verify("SKU-A 最高温度5℃。", rows).passed


@pytest.mark.parametrize(
    "answer",
    [
        "SKU-A 净重不是500g",
        "SKU-A 净重大于500g",
        "SKU-A 净重<500g",
        "SKU-A 净重八百克",
        "SKU-A 净重−500g",
        "XYZ999 净重500g",
    ],
)
def test_unsupported_relationships_numerals_and_unknown_identifiers_cannot_pass(answer):
    result = SKUVerifier().verify(answer, [{"SKU": "SKU-A", "净重": "0.5kg"}], query="SKU-A净重")
    assert not result.passed


def test_length_dimensions_keep_every_axis_and_support_equivalent_conversion():
    rows = [{"SKU": "SKU-A", "外箱尺寸": "260×210×280mm", "电压": "220V", "频率": "50Hz"}]
    verifier = SKUVerifier()
    assert verifier.verify("SKU-A 外箱尺寸26×21×28cm，电压0.22kV，频率0.05kHz", rows).passed
    assert not verifier.verify("SKU-A 外箱尺寸999×210×280mm", rows).passed
    assert not verifier.verify("SKU-A 外箱尺寸280mm", rows).passed


def test_untrusted_summary_claim_without_mysql_truth_is_rejected():
    result = SKUVerifier().verify("旧摘要说 SKU-A 净重500g。", [], query="SKU-A净重？")
    assert not result.passed


def test_conflicting_duplicate_sku_across_rows_requires_both_truths():
    rows = [{"SKU": "SKU-A", "净重": "0.5kg"}, {"SKU": "SKU-A", "净重": "0.8kg"}]
    assert not SKUVerifier().verify("净重500g", rows, query="SKU-A净重？").passed


def test_quantity_only_answer_with_unknown_unit_is_not_treated_as_no_claim():
    verifier = SKUVerifier()
    rows = [{"SKU": "SKU-A", "净重": "0.5kg"}]
    assert not verifier.verify("500lb", rows, query="SKU-A？").passed
    assert not verifier.verify("500lb", []).passed
    assert not verifier.verify("净重500g", rows, query="ITEM-X的净重？").passed


def test_control_temperature_is_not_environment_working_temperature():
    rows = [{"SKU": "SKU-A", "控温范围": "35–100℃"}]
    verifier = SKUVerifier()
    assert verifier.verify("SKU-A 控温范围35–100℃", rows).passed
    assert not verifier.verify("SKU-A 工作温度35–100℃", rows).passed


def test_multi_sku_identity_survives_attribute_separator():
    """参数列使用分号排版时仍绑定最近明确的型号，不能串用另一商品真值。"""
    rows = [{"SKU": "ITEM-A", "净重": "0.5 kg"}, {"SKU": "ITEM-B", "净重": "0.8 kg"}]
    verifier = SKUVerifier()
    assert verifier.verify(
        "SKU：ITEM-A；净重500g。SKU：ITEM-B；净重800g。", rows, query="ITEM-A与ITEM-B净重"
    ).passed
    assert not verifier.verify(
        "SKU：ITEM-A；净重800g。SKU：ITEM-B；净重500g。", rows, query="ITEM-A与ITEM-B净重"
    ).passed


def test_sop_physical_parameter_requires_verbatim_authorized_context():
    """SOP 纸宽不冒充商品参数，只有带完整上下文的证据原句才可通过。"""
    source = "核对纸卷与模板匹配：平台甲标准面单宽度为 76 mm。"
    verifier = SKUVerifier()
    assert verifier.verify(source, [], query="打印机怎么排障", evidence_texts=[source]).passed
    assert not verifier.verify(
        source.replace("76", "90"), [], query="打印机怎么排障", evidence_texts=[source]
    ).passed
    assert not verifier.verify("76 mm。", [], evidence_texts=[source]).passed
