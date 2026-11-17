"""离线证据合成：仅抽取授权上下文，按章节、对象、表格行和数值条件组织答案。"""

import json
import re
from collections import defaultdict
from decimal import Decimal
from typing import Any

from aegis.guardrails.access_policy import SAFE_EVIDENCE_REFUSAL

NO_EVIDENCE = SAFE_EVIDENCE_REFUSAL
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9_-])"
)
_MONEY = re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>亿|万)?\s*元")
_FIELDS = (
    ("净重", "净质量"),
    ("包装毛重", "毛重"),
    ("容量", "容积"),
    ("额定功率", "标称功率"),
    ("最大功率", "峰值功率"),
    ("控温范围", "工作温度范围"),
    ("温控精度", "温度精度"),
    ("温度上限", "最高出水温度", "最高温度"),
    ("存储温度", "储存温度"),
)
_STOP = {
    "什么",
    "怎么",
    "多少",
    "如何",
    "这个",
    "我们",
    "客户",
    "告诉",
    "一下",
    "可以",
    "是否",
    "需要",
    "不能",
    "应该",
    "是多少",
}

# 这些是问法同义词，不包含权限身份、具体型号、金额或某份文档的事实值。
_TOPICS = (
    ("返利", "返点"),
    ("毛利",),
    ("坑位费", "保底 roi", "净 gmv"),
    ("坏账", "核销"),
    ("审批链", "审批权限", "应急垫付"),
    ("净重", "净质量"),
    ("容量", "容积"),
    ("功率",),
    ("滤芯",),
    ("延迟发货",),
    ("无理由",),
    ("碎裂", "破损"),
    ("乱码", "打印"),
    ("跳白纸",),
    ("卡单",),
    ("库位",),
    ("拦截", "拒收"),
)


def evidence_is_relevant(query: str, evidence: list[dict], *, keyword_only: bool = False) -> bool:
    """检查核心主题能否在证据中找到；常见部门词与金额重合不能替代主题依据。"""
    text = "\n".join(
        _body(item.get("content_text", "")) + json.dumps(_rows(item), ensure_ascii=False)
        for item in evidence
        if isinstance(item, dict)
    ).lower()
    question = query.lower()
    requested = [group for group in _TOPICS if any(term in question for term in group)]
    if requested:
        return all(any(term in text for term in group) for group in requested)
    if keyword_only:
        # 真实模型能够处理同义表达；未识别主题不等于已证明缺少证据。
        return True
    terms = _terms(query)
    matched = terms & _terms(text)
    return bool(matched) and len(matched) / max(len(terms), 1) >= 0.12


def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9_-]+", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms - _STOP


def _score(query: str, text: str) -> float:
    relevant = _terms(query)
    overlap = relevant & _terms(text)
    return sum(8 if "-" in term else 1 for term in overlap)


def _body(text: str) -> str:
    """去掉层级标题和围栏标记，避免根标题的关键词污染所有父块。"""
    return "\n".join(
        line for line in text.splitlines() if not line.startswith(("#", "```"))
    ).strip()


def _rows(item: dict) -> list[dict[str, Any]]:
    value = item.get("sku_rows_json") or {}
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        rows = [row for row in value["rows"] if isinstance(row, dict)]
        header = next(
            (
                line
                for line in item.get("content_text", "").splitlines()
                if line.strip().startswith("|")
            ),
            "",
        )
        columns = [cell.strip() for cell in header.strip().strip("|").split("|")]
        # MySQL JSON 会重排对象键；表格第一列的身份语义必须从原始表头恢复。
        return [{**{key: row[key] for key in columns if key in row}, **row} for row in rows]
    return []


def _row_identity(row: dict) -> str:
    for key, value in row.items():
        if "sku" in key.lower() or key in {"商品型号", "型号", "配件编码", "报错代码"}:
            return str(value)
    return ""


def _contains_identifier(text: str, identity: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(identity)}(?![A-Za-z0-9_-])", text, re.I))


def _requested_fields(query: str, row: dict) -> list[str]:
    requested = []
    for key in row:
        aliases = next((group for group in _FIELDS if key in group), ())
        if any(alias in query for alias in aliases):
            requested.append(key)
    return requested


def _render_row(row: dict, fields: list[str] | None = None) -> str:
    chosen = row if fields is None else {key: row[key] for key in fields}
    return "；".join(f"{key}：{value}" for key, value in chosen.items()) + "。"


def _amount(text: str) -> Decimal | None:
    match = _MONEY.search(text)
    if not match:
        return None
    factor = {None: Decimal(1), "万": Decimal(10000), "亿": Decimal(100000000)}[match["unit"]]
    return Decimal(match["number"]) * factor


def _range_matches(text: str, amount: Decimal) -> bool:
    matches = list(_MONEY.finditer(text))
    values = [_amount(match[0]) for match in matches]
    if not values:
        return False
    if len(values) >= 2:
        lower_inclusive = not text.startswith(("超过", "大于"))
        upper_inclusive = "不满" not in text[matches[0].end() :]
        return (amount >= values[0] if lower_inclusive else amount > values[0]) and (
            amount <= values[1] if upper_inclusive else amount < values[1]
        )
    if any(word in text for word in ("以下", "不满", "小于", "不超过")):
        return amount < values[0] if "不满" in text or "小于" in text else amount <= values[0]
    if any(word in text for word in ("以上", "超过", "不少于", "大于")):
        return amount > values[0] if "超过" in text or "大于" in text else amount >= values[0]
    return amount == values[0]


def _select_rows(query: str, rows: list[dict]) -> list[dict]:
    identities = [
        row
        for row in rows
        if _row_identity(row) and _contains_identifier(query, _row_identity(row))
    ]
    if identities:
        return identities
    compatible = [
        row
        for row in rows
        if any(
            any(_contains_identifier(query, code) for code in _IDENTIFIER.findall(str(value)))
            for key, value in row.items()
            if "适配" in key or "兼容" in key
        )
    ]
    if compatible:
        return compatible
    amount = _amount(query)
    ranged = [
        row
        for row in rows
        if amount is not None and _range_matches(str(next(iter(row.values()))), amount)
    ]
    if ranged:
        return ranged
    scores = [(_score(query, str(next(iter(row.values())))), row) for row in rows]
    highest = max((score for score, _ in scores), default=0)
    return [row for score, row in scores if score == highest] if highest >= 1 else []


def _plain_lines(item: dict, query: str) -> list[str]:
    """抽取正文而非表格或槽位模板；显式报错码问题不混入其他代码的动作。"""
    identifiers = set(_IDENTIFIER.findall(query))
    lines = []
    skip_slots = False
    for raw in _body(item.get("content_text", "")).splitlines():
        line = raw.strip()
        if not line or line.startswith("|"):
            continue
        if "【排障必填" in line or "【适用场景】" in line:
            skip_slots = True
            continue
        if line.startswith("**【"):
            skip_slots = False
        if skip_slots:
            continue
        line = re.sub(r"^\d+[.、]\s*", "", line)
        line = re.sub(r"\*\*【[^】]+】\*\*", "", line).strip()
        if not line:
            continue
        lead_code = re.match(r"([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)[:：]", line)
        if (
            lead_code
            and identifiers
            and not any(code.lower() == lead_code[1].lower() for code in identifiers)
        ):
            continue
        line_codes = set(_IDENTIFIER.findall(line))
        if identifiers and line_codes:
            compatible = {
                code
                for code in line_codes
                if any(
                    target.lower() == code.lower() or target.lower().startswith(code.lower() + "-")
                    for target in identifiers
                )
            }
            if not compatible and re.search(r"(?:报|提示)\s*[A-Za-z][A-Za-z0-9_-]+", line):
                continue
            if not compatible and _IDENTIFIER.match(line):
                continue
            if not compatible and not _IDENTIFIER.match(line):
                first = min(line.index(code) for code in line_codes)
                prefix = line[:first].rstrip("，；（( ")
                if prefix and "保修" in prefix:
                    line = prefix + "。"
        lines.append(line)
    return lines


def _sku_answer(query: str, evidence: list[dict]) -> str | None:
    rows = [row for item in evidence for row in _rows(item)]
    matched = [
        row
        for row in rows
        if any("sku" in key.lower() for key in row)
        and _contains_identifier(query, _row_identity(row))
    ]
    unique = {json.dumps(row, ensure_ascii=False, sort_keys=True): row for row in matched}
    matched = list(unique.values())
    if not matched:
        return None
    if any(
        key in query
        and not any(key in group for group in _FIELDS)
        and key not in {"SKU", "SKU 编码", "商品型号", "型号", "备注"}
        for row in matched
        for key in row
    ):
        return NO_EVIDENCE
    fields = [_requested_fields(query, row) for row in matched]
    fields = [
        [key for key in keys if not ("sku" in key.lower() or "型号" in key)] for keys in fields
    ]
    if not any(fields):
        if any(word in query for word in ("规格", "参数", "介绍", "信息")):
            fields = [
                [key for key in row if any(key in group for group in _FIELDS)] for row in matched
            ]
        else:
            return None
    parts = []
    for row, requested in zip(matched, fields, strict=True):
        identity_key = next(key for key in row if "sku" in key.lower())
        # 每个参数句完整携带商品身份，分号不会让多型号问句丢失行归属。
        parts.extend(
            f"{identity_key}：{row[identity_key]}，{key}：{row[key]}。" for key in requested
        )
        if any(word in query for word in ("区分", "区别", "插座", "用电", "要求")):
            remarks = str(row.get("备注", ""))
            pieces = re.split(r"[；;]", remarks)
            for piece in pieces:
                if piece and (
                    ("插座" in query and "插座" in piece)
                    or ("插座" not in query and not re.search(r"\d", piece))
                ):
                    parts.append(f"{row[identity_key]}：{piece}。")
    if any(word in query for word in ("区分", "区别")):
        for item in evidence:
            for line in _plain_lines(item, query):
                if not any(_contains_identifier(line, _row_identity(row)) for row in matched):
                    continue
                for phrase in re.split(r"[，。；]|且", line):
                    if any(word in phrase for word in ("铭牌", "标识")) and not re.search(
                        r"\d", phrase
                    ):
                        for row in matched:
                            identity = _row_identity(row)
                            suffix = identity.rsplit("-", 1)[-1]
                            phrase = re.sub(
                                rf"(?<![A-Za-z0-9_-]){re.escape(suffix)}(?![A-Za-z0-9_-])",
                                identity,
                                phrase,
                            )
                        parts.append(phrase + "。")
    if any(word in query for word in ("混用", "兼容", "混装")):
        for item in evidence:
            for line in _plain_lines(item, query):
                if any(_contains_identifier(line, _row_identity(row)) for row in matched):
                    for phrase in re.split(r"[，。；]", line):
                        if any(word in phrase for word in ("不兼容", "互不兼容", "严禁混装")):
                            parts.append(phrase + "。")
    return "\n".join(dict.fromkeys(parts))


def _calculate(query: str, rows: list[dict]) -> str | None:
    """仅使用已选中证据行中的比例和上下限，计算任意订单金额而非匹配示例金额。"""
    amount = _amount(query)
    if amount is None:
        return None
    for row in rows:
        ratio_key = next((key for key in row if "赔付比例" in key), None)
        lower_key = next((key for key in row if "下限" in key), None)
        upper_key = next((key for key in row if "上限" in key), None)
        if not ratio_key or not lower_key or not upper_key:
            continue
        ratio = re.search(r"(\d+(?:\.\d+)?)\s*%", str(row[ratio_key]))
        lower, upper = _amount(str(row[lower_key])), _amount(str(row[upper_key]))
        if ratio and lower is not None and upper is not None:
            raw = amount * Decimal(ratio[1]) / 100
            payable = min(max(raw, lower), upper)
            qualifier = (
                "低于下限" if raw < lower else "超过上限" if raw > upper else "处于上下限区间"
            )
            return (
                f"{amount:g} 元 × {ratio[1]}% = {raw:g} 元，{qualifier}，最终赔付 {payable:g} 元。"
            )
    return None


def synthesize_evidence(query: str, evidence: list[dict]) -> str:
    """匹配具体对象后按业务章节合成；拒答固定措辞不复述用户输入。"""
    evidence = [
        dict(item, content_text=item.get("content_text", ""))
        for item in evidence
        if isinstance(item, dict) and (isinstance(item.get("content_text"), str) or _rows(item))
    ]
    if not evidence:
        return NO_EVIDENCE
    if not evidence_is_relevant(query, evidence):
        return NO_EVIDENCE
    product = _sku_answer(query, evidence)
    if product:
        return product
    groups = defaultdict(list)
    for item in evidence:
        path = item.get("header_path", "")
        hierarchy = path.split(" > ") if path else []
        if len(hierarchy) < 2:
            headings = re.findall(r"^## (.+)$", item["content_text"], re.M)
            group = headings[0] if headings else path
        else:
            group = hierarchy[1]
        groups[(item.get("doc_code", ""), group)].append(item)
    ranked = []
    for key, items in groups.items():
        score = max(
            (
                _score(
                    query, _body(item["content_text"]) + json.dumps(_rows(item), ensure_ascii=False)
                )
                for item in items
            ),
            default=0,
        )
        score += 3 * _score(query, key[1])
        if _amount(query) is not None and any(_select_rows(query, _rows(item)) for item in items):
            score += 3
        ranked.append((score, key, items))
    score, _, selected = max(ranked, key=lambda value: value[0])
    if score < 2:
        return NO_EVIDENCE
    answer = []
    selected_rows = []
    for item in selected:
        leaf = item.get("header_path", "").split(" > ")[-1]
        if "示例" in leaf and _amount(query) is None and "示例" not in query:
            continue
        if (
            any(word in leaf for word in ("特殊", "大促", "体验"))
            and not any(word in query for word in ("特殊", "大促", "体验", "双", "活动"))
            and not any(
                _contains_identifier(query, code)
                for code in _IDENTIFIER.findall(_body(item["content_text"]))
            )
        ):
            continue
        # 同一大节下的表格、公式与追偿可属不同主题，不把兄弟子节全部倾倒给调用方。
        if "毛利" in query and any(word in leaf for word in ("核算公式", "保底", "追偿")):
            continue
        rows = _rows(item)
        chosen = _select_rows(query, rows)
        selected_rows.extend(chosen)
        answer.extend(_render_row(row) for row in chosen)
        lines = _plain_lines(item, query)
        # 示例只保留输入金额对应的条目；其他业务规则仍来自原文段落。
        for line in lines:
            if "实付" in line and "×" in line and _amount(query) is not None:
                if _amount(line) != _amount(query):
                    continue
            answer.append(line)
    calculation = _calculate(query, selected_rows)
    if calculation:
        answer.insert(0, calculation)
    if "小数" in query:
        for line in list(answer):
            for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", line):
                answer.append(f"{match[1]}% 等价于小数 {Decimal(match[1]) / 100:g}。")
    return "\n".join(dict.fromkeys(answer))[:12000] if answer else NO_EVIDENCE
