"""按标题构建父子块，并在任何切片边界保留完整的 SKU 表格行。"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aegis.db.models import DocumentType

_HEADING = re.compile(r"^ {0,3}(#{1,3})[ \t]+([^\r\n]*?)[ \t]*(?:\r?\n)?$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_SEPARATOR = re.compile(r"^:?-{3,}:?$")


class DocumentMetadata(BaseModel):
    """入库前校验文档身份与权限，空部门不具有隐式公开含义。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    doc_code: str = Field(min_length=1, max_length=64)
    doc_type: DocumentType
    dept_scope: str = Field(min_length=1, max_length=64)
    min_role_level: int = Field(default=1, ge=1, strict=True)

    @field_validator("doc_code", "dept_scope")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 for char in value):
            raise ValueError("文档编码和部门不能包含控制字符")
        return value


class ParentChunk(BaseModel):
    """用于 MySQL 回溯的完整父块；结构化表格只包含本块的实际数据行。"""

    model_config = ConfigDict(frozen=True)

    parent_id: str
    header_path: str
    content_text: str
    is_sku_table: bool = False
    sku_rows_json: dict[str, Any] | None = None


class ChildChunk(BaseModel):
    """用于双路召回的子块，UUID 可直接作为 Qdrant 点标识。"""

    model_config = ConfigDict(frozen=True)

    child_id: str
    parent_id: str
    header_path: str
    content_text: str
    is_sku_table: bool = False
    sku_rows_json: dict[str, Any] | None = None


class ParsedDocument(BaseModel):
    """原文校验和与父子索引的一次确定性解析结果。"""

    model_config = ConfigDict(frozen=True)

    metadata: DocumentMetadata
    sha256_hash: str
    title: str
    parents: list[ParentChunk]
    children: list[ChildChunk]


@dataclass
class _Table:
    header: str
    keys: list[str]
    lines: list[str]
    rows: list[dict[str, str]]


@dataclass
class _Block:
    text: str = ""
    table: _Table | None = None
    # 围栏代码块整体保存，代码里的标题和管道符不参与 Markdown 分段。
    fenced: bool = False

    @property
    def rendered(self) -> str:
        if self.table is not None:
            return self.table.header + "".join(self.table.lines)
        return self.text


@dataclass
class _Section:
    headings: tuple[tuple[int, str], ...]
    blocks: list[_Block] = field(default_factory=list)

    @property
    def prefix(self) -> str:
        return "".join(f"{'#' * level} {title}\n" for level, title in self.headings)

    @property
    def path(self) -> str:
        return " > ".join(title for _, title in self.headings)


def _cells(line: str) -> list[str] | None:
    """识别转义管道和行内代码，避免把一个单元格误拆成多列。"""
    value = line.strip()
    cells: list[str] = []
    current: list[str] = []
    tick_width = 0
    pipe_found = False
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and index + 1 < len(value):
            # 结构化值保留字面内容，Markdown 原始行另行原样保存。
            following = value[index + 1]
            if following in "|\\`":
                current.append(following)
                index += 2
                continue
        if char == "`":
            end = index + 1
            while end < len(value) and value[end] == "`":
                end += 1
            width = end - index
            if tick_width == 0:
                tick_width = width
            elif tick_width == width:
                tick_width = 0
            current.append(value[index:end])
            index = end
            continue
        if char == "|" and tick_width == 0:
            pipe_found = True
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    if not pipe_found:
        return None
    if value.startswith("|"):
        cells.pop(0)
    # 末尾转义的管道符属于单元格，不能剔除。
    if cells and cells[-1] == "" and value.endswith("|"):
        cells.pop()
    return cells


def _table_keys(headers: list[str]) -> list[str]:
    """空列名和重复列名使用稳定后缀，不能静默覆盖 SKU 参数。"""
    used: set[str] = set()
    result = []
    for index, header in enumerate(headers, 1):
        base = header or f"列{index}"
        key, suffix = base, 2
        while key in used:
            key = f"{base}__{suffix}"
            suffix += 1
        used.add(key)
        result.append(key)
    return result


def _read_table(lines: list[str], start: int) -> tuple[_Table, int] | None:
    if start + 1 >= len(lines):
        return None
    headers = _cells(lines[start])
    separators = _cells(lines[start + 1])
    if (
        not headers
        or not separators
        or len(headers) != len(separators)
        or not all(_SEPARATOR.fullmatch(cell) for cell in separators)
    ):
        return None
    keys = _table_keys(headers)
    table_lines, rows = [], []
    index = start + 2
    while index < len(lines):
        if _HEADING.match(lines[index]) or _FENCE.match(lines[index]):
            break
        values = _cells(lines[index])
        if values is None or not lines[index].strip():
            break
        # 多余列不能丢弃；缺失列明确填为空字符串，保留可追溯的原始行。
        row_keys = keys.copy()
        if len(values) > len(keys):
            row_keys = _table_keys(
                keys + [f"额外列{i + 1}" for i in range(len(values) - len(keys))]
            )
        rows.append(dict(zip(row_keys, values + [""] * (len(row_keys) - len(values)), strict=True)))
        table_lines.append(lines[index])
        index += 1
    return _Table("".join(lines[start : start + 2]), keys, table_lines, rows), index


def _sections(text: str) -> list[_Section]:
    """仅 H1 到 H3 改变语义路径；更深标题作为当前段落内容保留。"""
    lines = text.splitlines(keepends=True)
    sections: list[_Section] = []
    section = _Section(())
    headings: list[tuple[int, str]] = []
    plain: list[str] = []

    def flush_plain() -> None:
        if plain:
            section.blocks.append(_Block(text="".join(plain)))
            plain.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _FENCE.match(line)
        if fence:
            flush_plain()
            marker = fence.group(1)
            end = index + 1
            closing = re.compile(rf"^ {{0,3}}{re.escape(marker[0])}{{{len(marker)},}}[ \t]*$")
            while end < len(lines):
                if closing.fullmatch(lines[end].rstrip("\r\n")):
                    end += 1
                    break
                end += 1
            section.blocks.append(_Block(text="".join(lines[index:end]), fenced=True))
            index = end
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_plain()
            if section.blocks or section.headings:
                sections.append(section)
            level = len(heading.group(1))
            headings = [(depth, title) for depth, title in headings if depth < level]
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(2))
            headings.append((level, title))
            section = _Section(tuple(headings))
            index += 1
            continue
        table_result = _read_table(lines, index)
        if table_result is not None:
            flush_plain()
            table, index = table_result
            section.blocks.append(_Block(table=table))
            continue
        plain.append(line)
        index += 1
    flush_plain()
    if section.blocks or section.headings:
        sections.append(section)
    return sections


def _table_fragment(table: _Table, start: int, end: int) -> _Block:
    return _Block(
        table=_Table(table.header, table.keys, table.lines[start:end], table.rows[start:end])
    )


def _table_chunks(table: _Table, size: int) -> list[_Block]:
    """表头重复注入每片；超长单行优先保持完整，允许突破目标字符数。"""
    if not table.lines:
        return [_table_fragment(table, 0, 0)]
    chunks: list[_Block] = []
    start, length = 0, len(table.header)
    for index, line in enumerate(table.lines):
        if index > start and length + len(line) > size:
            chunks.append(_table_fragment(table, start, index))
            start, length = index, len(table.header)
        length += len(line)
    chunks.append(_table_fragment(table, start, len(table.lines)))
    return chunks


def _parent_groups(section: _Section, size: int) -> list[list[_Block]]:
    budget = max(1, size - len(section.prefix))
    groups: list[list[_Block]] = []
    current: list[_Block] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        if current:
            groups.append(current)
            current, length = [], 0

    for block in section.blocks:
        if block.table is not None:
            for piece in _table_chunks(block.table, budget):
                if length + len(piece.rendered) > budget:
                    flush()
                current.append(piece)
                length += len(piece.rendered)
            continue
        if block.fenced:
            if length + len(block.text) > budget:
                flush()
            current.append(block)
            length += len(block.text)
            continue
        remaining = block.text
        while remaining:
            if length >= budget:
                flush()
            take = min(budget - length, len(remaining))
            current.append(_Block(text=remaining[:take]))
            length += take
            remaining = remaining[take:]
    flush()
    # 没有正文的标题也保留，不能把文档声明悄悄丢弃。
    return groups or ([[]] if section.headings else [])


def _row_payload(blocks: list[_Block]) -> dict[str, Any] | None:
    tables = [block.table for block in blocks if block.table is not None]
    return {"rows": [row for table in tables for row in table.rows]} if tables else None


class MarkdownParser:
    """标题语义分段后的父块约 2000 字符，普通子块约 400 字符并重叠 50。

    标题路径与表头计入目标长度。完整表格行、围栏代码块和过长标题具有更高
    优先级，因此目标长度不是截断上限；表格子块不以字符重叠制造残缺行。
    """

    def __init__(self, parent_size: int = 2000, child_size: int = 400, overlap: int = 50):
        if not 0 <= overlap < child_size <= parent_size:
            raise ValueError("必须满足 0 <= overlap < child_size <= parent_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def parse(self, text: str, metadata: DocumentMetadata | dict[str, Any]) -> ParsedDocument:
        """校验和使用未经归一化的原始 UTF-8 内容，权限变化也会生成新块标识。"""
        metadata = DocumentMetadata.model_validate(metadata)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = json.dumps(
            {
                "metadata": metadata.model_dump(mode="json"),
                "sha256_hash": digest,
                "parser_version": 1,
                "parent_size": self.parent_size,
                "child_size": self.child_size,
                "overlap": self.overlap,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        sections = _sections(text)
        title = next(
            (title for section in sections for level, title in section.headings if level == 1),
            metadata.doc_code,
        )
        if len(title) > 255:
            raise ValueError("文档主标题不能超过 255 字符")
        for section in sections:
            if len(section.path) > 512:
                raise ValueError("H1-H3 标题路径不能超过 512 字符")
            for group in _parent_groups(section, self.parent_size):
                parent_id = uuid5(NAMESPACE_URL, f"{identity}:parent:{len(parents)}").hex
                payload = _row_payload(group)
                parent = ParentChunk(
                    parent_id=parent_id,
                    header_path=section.path,
                    content_text=section.prefix + "".join(block.rendered for block in group),
                    is_sku_table=payload is not None,
                    sku_rows_json=payload,
                )
                parents.append(parent)
                children.extend(self._children(parent, section.prefix, group))
        return ParsedDocument(
            metadata=metadata,
            sha256_hash=digest,
            title=title,
            parents=parents,
            children=children,
        )

    def _children(self, parent: ParentChunk, prefix: str, blocks: list[_Block]) -> list[ChildChunk]:
        budget = max(1, self.child_size - len(prefix))
        # 标题可能占满目标长度；这种情况下减少重叠以保证游标前进。
        overlap = min(self.overlap, budget - 1)
        fragments: list[list[_Block]] = []
        prose = ""

        def flush_prose() -> None:
            nonlocal prose
            start = 0
            while start < len(prose):
                end = min(start + budget, len(prose))
                fragments.append([_Block(text=prose[start:end])])
                if end == len(prose):
                    break
                start = end - overlap
            prose = ""

        for block in blocks:
            if block.table is not None:
                flush_prose()
                fragments.extend([piece] for piece in _table_chunks(block.table, budget))
            elif block.fenced:
                flush_prose()
                fragments.append([block])
            else:
                prose += block.text
        flush_prose()
        if not fragments:
            fragments.append([])
        result = []
        for index, fragment in enumerate(fragments):
            payload = _row_payload(fragment)
            result.append(
                ChildChunk(
                    child_id=str(uuid5(NAMESPACE_URL, f"{parent.parent_id}:child:{index}")),
                    parent_id=parent.parent_id,
                    header_path=parent.header_path,
                    content_text=prefix + "".join(block.rendered for block in fragment),
                    is_sku_table=payload is not None,
                    sku_rows_json=payload,
                )
            )
        return result
