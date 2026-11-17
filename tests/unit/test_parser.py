"""验证切片边界、标题上下文、SKU 原始行与结构化字段之间的一致性。"""

import hashlib
from uuid import UUID

import pytest
from pydantic import ValidationError

from aegis.db.models import DocumentType
from aegis.rag.parser import DocumentMetadata, MarkdownParser


@pytest.fixture
def metadata():
    return DocumentMetadata(
        doc_code="SKU-THERMO-2026",
        doc_type=DocumentType.SKU_SPEC,
        dept_scope="warehouse",
        min_role_level=2,
    )


def test_parent_and_child_sizes_overlap_and_full_text_coverage(metadata):
    # 每个字符都能独立定位，重复字符不会掩盖边界丢字或错误重叠。
    text = "".join(chr(0x4E00 + index) for index in range(4500))
    parsed = MarkdownParser().parse(text, metadata)
    assert [len(parent.content_text) for parent in parsed.parents] == [2000, 2000, 500]
    assert "".join(parent.content_text for parent in parsed.parents) == text
    for parent in parsed.parents:
        children = [child for child in parsed.children if child.parent_id == parent.parent_id]
        assert all(len(child.content_text) <= 400 for child in children)
        restored = children[0].content_text
        for previous, current in zip(children, children[1:], strict=False):
            assert previous.content_text[-50:] == current.content_text[:50]
            restored += current.content_text[50:]
        assert restored == parent.content_text


def test_semantic_h1_h2_h3_paths_and_deeper_heading_preservation(metadata):
    text = (
        "前言说明\n# 旗舰系列\n系列通用说明。\n"
        "## A 型\nA 型参数。\n### 加热\n额定 1500 W。\n#### 警告\n禁止空烧。\n"
        "## B 型\nB 型参数。\n# 售后\n售后电话分机 800。\n"
    )
    parsed = MarkdownParser().parse(text, metadata)
    assert parsed.title == "旗舰系列"
    assert [parent.header_path for parent in parsed.parents] == [
        "",
        "旗舰系列",
        "旗舰系列 > A 型",
        "旗舰系列 > A 型 > 加热",
        "旗舰系列 > B 型",
        "售后",
    ]
    child = next(child for child in parsed.children if "额定 1500 W" in child.content_text)
    assert child.content_text.startswith("# 旗舰系列\n## A 型\n### 加热\n")
    assert "#### 警告\n禁止空烧。" in child.content_text
    assert all(
        "A 型" not in child.content_text
        for child in parsed.children
        if "B 型参数" in child.content_text
    )


def test_headers_count_towards_target_size_and_are_injected_into_every_child(metadata):
    prefix = "# 同系列商品\n## 规格\n### 重量\n"
    body = "额定净重为 2.5 kg，包装重量为 3.1 kg。" * 220
    parsed = MarkdownParser().parse(prefix + body, metadata)
    parents = [parent for parent in parsed.parents if parent.header_path.endswith("重量")]
    assert len(parents) > 1
    assert all(len(parent.content_text) <= 2000 for parent in parents)
    assert "".join(parent.content_text.removeprefix(prefix) for parent in parents) == body
    children = [child for child in parsed.children if child.header_path.endswith("重量")]
    assert all(child.content_text.startswith(prefix) for child in children)
    assert all(len(child.content_text) <= 400 for child in children)


def test_large_sku_table_has_complete_rows_repeated_headers_and_structured_payload(metadata):
    heading = "# 智能热水系列\n## A 系列\n### SKU 对照\n"
    header = "| SKU | 净重 | 额定功率 | 工作温度 | 区别说明 |\n"
    separator = "| --- | ---: | ---: | --- | --- |\n"
    rows = [
        f"| TH-A-{index:03d} | {index + 1}.5 kg | {1000 + index} W | "
        f"5~95 ℃ | 仅限型号 {index:03d} |\n"
        for index in range(90)
    ]
    parsed = MarkdownParser().parse(heading + header + separator + "".join(rows), metadata)
    parents = [parent for parent in parsed.parents if parent.is_sku_table]
    children = [child for child in parsed.children if child.is_sku_table]
    assert len(parents) >= 3
    assert len(children) > len(parents)
    for chunk in [*parents, *children]:
        assert chunk.content_text.startswith(heading + header + separator)
        assert chunk.sku_rows_json is not None
        source_rows = [line for line in chunk.content_text.splitlines() if line.startswith("| TH-")]
        assert len(source_rows) == len(chunk.sku_rows_json["rows"])
        assert all(line + "\n" in rows for line in source_rows)
        assert all(line.endswith("|") for line in source_rows)
    parent_rows = [row for parent in parents for row in parent.sku_rows_json["rows"]]
    child_rows = [row for child in children for row in child.sku_rows_json["rows"]]
    assert parent_rows == child_rows
    assert len(parent_rows) == 90
    assert parent_rows[0] == {
        "SKU": "TH-A-000",
        "净重": "1.5 kg",
        "额定功率": "1000 W",
        "工作温度": "5~95 ℃",
        "区别说明": "仅限型号 000",
    }
    assert all(len(chunk.content_text) <= 2000 for chunk in parents)
    assert all(len(chunk.content_text) <= 400 for chunk in children)


def test_single_oversized_table_row_is_never_cut(metadata):
    row = "| TH-X | " + "此型号具有独立温控与超温保护。" * 200 + " | 2200 W |\n"
    text = "# 型号\n| SKU | 说明 | 功率 |\n| --- | --- | --- |\n" + row
    parsed = MarkdownParser().parse(text, metadata)
    parent = next(parent for parent in parsed.parents if parent.is_sku_table)
    child = next(child for child in parsed.children if child.is_sku_table)
    assert row in parent.content_text and row in child.content_text
    assert len(child.content_text) > 2000
    assert child.sku_rows_json["rows"][0]["功率"] == "2200 W"


def test_escaped_pipes_inline_code_empty_and_duplicate_table_headers(metadata):
    text = (
        "SKU | 参数 | 参数 |\n"
        "--- | :---: | ---: |\n"
        "A | 红\\|蓝 | `a|b` |\n"
        "B | 铜 |\n"
        "C | 铜 | 银 | 多余值 |\n\n"
        "| | 重量 |\n| --- | --- |\n| D | 2 kg |\n"
    )
    parsed = MarkdownParser().parse(text, metadata)
    rows = [row for parent in parsed.parents for row in parent.sku_rows_json["rows"]]
    assert rows == [
        {"SKU": "A", "参数": "红|蓝", "参数__2": "`a|b`"},
        {"SKU": "B", "参数": "铜", "参数__2": ""},
        {"SKU": "C", "参数": "铜", "参数__2": "银", "额外列1": "多余值"},
        {"列1": "D", "重量": "2 kg"},
    ]
    assert "红\\|蓝" in parsed.parents[0].content_text


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fenced_markdown_is_not_parsed_as_headings_or_sku_tables(metadata, fence):
    fenced = f"{fence}markdown\n# 伪标题\n| SKU | 重量 |\n| --- | --- |\n| A | 2 kg |\n{fence}\n"
    parsed = MarkdownParser().parse("# 操作指引\n" + fenced + "## 实际章节\n请扫码。", metadata)
    assert [parent.header_path for parent in parsed.parents] == ["操作指引", "操作指引 > 实际章节"]
    assert not any(parent.is_sku_table for parent in parsed.parents)
    assert any(fenced in child.content_text for child in parsed.children)


def test_unclosed_fence_preserves_remaining_document(metadata):
    text = "# 操作\n```text\n# 不是真标题\n不能丢失的内容"
    parsed = MarkdownParser().parse(text, metadata)
    assert len(parsed.parents) == 1
    assert parsed.parents[0].content_text == text
    assert not parsed.parents[0].is_sku_table


def test_invalid_table_separator_does_not_create_table(metadata):
    text = "| SKU | 重量 |\n| --- | 无分隔符 |\n| A | 2 kg |\n"
    parsed = MarkdownParser().parse(text, metadata)
    assert not parsed.parents[0].is_sku_table
    assert parsed.parents[0].sku_rows_json is None
    assert parsed.parents[0].content_text == text


def test_empty_table_preserves_header_and_empty_rows(metadata):
    text = "| SKU | 重量 |\n| --- | --- |\n"
    parsed = MarkdownParser().parse(text, metadata)
    assert parsed.parents[0].content_text == text
    assert parsed.parents[0].sku_rows_json == {"rows": []}


def test_checksum_and_ids_are_deterministic_but_change_with_content_acl_or_sizing(metadata):
    text = "# 型号规格\r\n净重 2.5 kg。\r\n"
    parser = MarkdownParser()
    first = parser.parse(text, metadata)
    assert first == parser.parse(text, metadata.model_dump())
    assert first.sha256_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert all(UUID(child.child_id) for child in first.children)
    changed_acl = parser.parse(text, metadata.model_copy(update={"min_role_level": 3}))
    changed_text = parser.parse(text.replace("2.5", "2.6"), metadata)
    changed_size = MarkdownParser(child_size=300).parse(text, metadata)
    assert first.sha256_hash == changed_acl.sha256_hash
    assert first.sha256_hash != changed_text.sha256_hash
    parent_ids = {
        item.parents[0].parent_id for item in [first, changed_acl, changed_text, changed_size]
    }
    assert len(parent_ids) == 4


def test_empty_document_and_heading_only_document(metadata):
    empty = MarkdownParser().parse("", metadata)
    assert empty.parents == [] and empty.children == []
    assert empty.title == metadata.doc_code
    parsed = MarkdownParser().parse("# 只有标题", metadata)
    assert len(parsed.parents) == len(parsed.children) == 1
    assert parsed.parents[0].content_text == "# 只有标题\n"


def test_literal_hash_in_heading_is_preserved_but_closing_marker_removed(metadata):
    parsed = MarkdownParser().parse("# SKU-C#\n参数\n## 重量 ###\n2 kg", metadata)
    assert parsed.title == "SKU-C#"
    assert parsed.parents[-1].header_path == "SKU-C# > 重量"


def test_heading_with_pipe_ends_a_preceding_table(metadata):
    parsed = MarkdownParser().parse(
        "| SKU | 重量 |\n| --- | --- |\n| A | 2 kg |\n## 后续 | 说明\n正文", metadata
    )
    assert parsed.parents[0].sku_rows_json == {"rows": [{"SKU": "A", "重量": "2 kg"}]}
    assert parsed.parents[-1].header_path == "后续 | 说明"


@pytest.mark.parametrize(
    "values",
    [
        {"parent_size": 100, "child_size": 400},
        {"child_size": 50, "overlap": 50},
        {"overlap": -1},
        {"child_size": 0},
    ],
)
def test_invalid_size_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        MarkdownParser(**values)


@pytest.mark.parametrize(
    "update",
    [
        {"dept_scope": " "},
        {"dept_scope": "warehouse\nfinance"},
        {"doc_code": ""},
        {"min_role_level": -1},
        {"min_role_level": True},
        {"doc_type": "UNKNOWN"},
    ],
)
def test_document_metadata_rejects_invalid_identity_or_acl(metadata, update):
    with pytest.raises(ValidationError):
        DocumentMetadata.model_validate(metadata.model_dump() | update)


def test_oversized_db_metadata_is_rejected_without_truncation(metadata):
    with pytest.raises(ValueError, match="255"):
        MarkdownParser().parse("# " + "标题" * 128, metadata)
    with pytest.raises(ValueError, match="512"):
        MarkdownParser().parse(
            "# " + "甲" * 200 + "\n## " + "乙" * 200 + "\n### " + "丙" * 200, metadata
        )
