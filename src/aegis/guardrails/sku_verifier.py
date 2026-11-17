"""按商品行和属性核对量纲，不能用别的 SKU 或字段中的同值替代真值。"""

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aegis.db.models import KBParentChunk

_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UNIT = (
    r"(?:kg|ml|kw|mm|cm|kv|mv|khz|mhz|hz|g|l|w|m|v|°\s*c|℃|"
    r"千克|公斤|毫升|克|升|千瓦|瓦|毫米|厘米|米|千伏|毫伏|伏|赫兹|摄氏度)"
)
_DIMENSIONS = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<values>{_NUMBER}(?:\s*[×x*]\s*{_NUMBER}){{1,2}})"
    r"\s*(?P<unit>mm|cm|m|毫米|厘米|米)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<tolerance>±)?\s*(?P<first>{_NUMBER})\s*"
    rf"(?:(?P<first_unit>{_UNIT})?\s*(?P<range>[-–—~～至到])\s*"
    rf"(?P<second>{_NUMBER})\s*)?(?P<unit>{_UNIT})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SUSPICIOUS = re.compile(rf"(?<![A-Za-z0-9_.]){_NUMBER}\s*(?:[A-Za-z°]+|磅|盎司)", re.IGNORECASE)
_UNITS = {
    "kg": ("mass", Decimal("1000")),
    "千克": ("mass", Decimal("1000")),
    "公斤": ("mass", Decimal("1000")),
    "g": ("mass", Decimal("1")),
    "克": ("mass", Decimal("1")),
    "l": ("volume", Decimal("1000")),
    "升": ("volume", Decimal("1000")),
    "ml": ("volume", Decimal("1")),
    "毫升": ("volume", Decimal("1")),
    "kw": ("power", Decimal("1000")),
    "千瓦": ("power", Decimal("1000")),
    "w": ("power", Decimal("1")),
    "瓦": ("power", Decimal("1")),
    "°c": ("temperature", Decimal("1")),
    "℃": ("temperature", Decimal("1")),
    "摄氏度": ("temperature", Decimal("1")),
    "mm": ("length", Decimal("1")),
    "毫米": ("length", Decimal("1")),
    "cm": ("length", Decimal("10")),
    "厘米": ("length", Decimal("10")),
    "m": ("length", Decimal("1000")),
    "米": ("length", Decimal("1000")),
    "v": ("voltage", Decimal("1")),
    "伏": ("voltage", Decimal("1")),
    "kv": ("voltage", Decimal("1000")),
    "千伏": ("voltage", Decimal("1000")),
    "mv": ("voltage", Decimal("0.001")),
    "毫伏": ("voltage", Decimal("0.001")),
    "hz": ("frequency", Decimal("1")),
    "赫兹": ("frequency", Decimal("1")),
    "khz": ("frequency", Decimal("1000")),
    "mhz": ("frequency", Decimal("1000000")),
}
# 具体属性单独分组；净重/毛重、额定/峰值、工作/存储温度不能合并。
_ATTRIBUTE_GROUPS = (
    ("净重", "净质量", "net weight"),
    ("包装毛重", "毛重", "gross weight"),
    ("容量", "容积", "capacity", "volume"),
    ("额定功率", "标称功率", "rated power"),
    ("最大功率", "峰值功率", "max power", "maximum power"),
    ("控温范围", "设定温度范围"),
    ("工作温度", "工作温度范围", "运行温度", "环境工作温度"),
    ("存储温度", "储存温度", "存储温度范围"),
    ("温控精度", "温度精度", "温控误差"),
    ("温度上限", "最高温度"),
    ("温度下限", "最低温度"),
    ("重量", "质量", "weight", "mass"),
    ("功率", "power"),
    ("温度", "temperature"),
)
_ID_KEYS = {"sku", "sku编码", "sku编号", "商品编码", "商品编号", "型号", "商品型号", "model"}


def _normalize(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
        if unicodedata.category(char) != "Cf"
    )


@dataclass(frozen=True)
class Quantity:
    """值统一为 g、ml、W、摄氏度；区间和公差与标量严格区分。"""

    dimension: str
    values: tuple[Decimal, ...]
    kind: str
    start: int
    end: int
    text: str


def extract_quantities(text: str) -> list[Quantity]:
    """提取单位、区间和正负温度，不使用浮点数比较规格真值。"""
    normalized = _normalize(text)
    result = []
    for match in _DIMENSIONS.finditer(normalized):
        dimension, factor = _UNITS[match["unit"]]
        values = tuple(
            Decimal(value.replace(",", "")) * factor
            for value in re.split(r"\s*[×x*]\s*", match["values"])
        )
        result.append(
            Quantity(dimension, values, "dimensions", match.start(), match.end(), match.group())
        )
    for match in _QUANTITY.finditer(normalized):
        if any(item.start <= match.start() and item.end >= match.end() for item in result):
            continue
        unit = re.sub(r"\s+", "", match["unit"])
        dimension, factor = _UNITS[unit]
        first_unit = re.sub(r"\s+", "", match["first_unit"] or unit)
        first_dimension, first_factor = _UNITS[first_unit]
        if first_dimension != dimension:
            # 不同量纲拼接的伪区间保留非法类型，永远不能匹配真实规格。
            dimension = "invalid"
        values = [Decimal(match["first"].replace(",", "")) * first_factor]
        if match["second"] is not None:
            values.append(Decimal(match["second"].replace(",", "")) * factor)
        kind = "range" if match["range"] else "tolerance" if match["tolerance"] else "scalar"
        if normalized[match.end() :].lstrip().startswith(("/", "²", "^")):
            dimension = "invalid"
        result.append(
            Quantity(dimension, tuple(values), kind, match.start(), match.end(), match.group())
        )
    return sorted(result, key=lambda item: item.start)


def _cell_text(value: Any) -> str:
    if isinstance(value, dict) and "数值" in value and "单位" in value:
        return f"{value['数值']} {value['单位']}"
    return str(value)


def _cell_quantities(key: str, value: Any) -> list[Quantity]:
    text = _cell_text(value)
    quantities = extract_quantities(text)
    if not quantities:
        # 兼容表头写单位、单元格仅写数值的常见 SKU 表。
        unit = re.search(rf"[（(\[]\s*({_UNIT})\s*[）)\]]", _normalize(key))
        if unit and re.fullmatch(_NUMBER, text.strip()):
            quantities = extract_quantities(f"{text} {unit[1]}")
    return quantities


def _aliases(key: str) -> set[str]:
    normalized = _normalize(key)
    clean = re.sub(r"[（(\[].*?[）)\]]", "", normalized).strip()
    result = {clean}
    # 仅按完整字段归组，避免“净重”落入泛化的“重量”组。
    for group in _ATTRIBUTE_GROUPS:
        if clean in group:
            result.update(group)
            break
    return result


def _identifiers(row: dict[str, Any]) -> list[str]:
    return [
        _normalize(str(value))
        for key, value in row.items()
        if _normalize(key).replace(" ", "") in _ID_KEYS and str(value).strip()
    ]


def _mentions(text: str, rows: Sequence[dict[str, Any]]) -> list[tuple[int, int, int]]:
    matches = []
    for index, row in enumerate(rows):
        for identifier in _identifiers(row):
            pattern = rf"(?<![A-Za-z0-9_-]){re.escape(identifier)}(?![A-Za-z0-9_-])"
            matches.extend(
                (match.start(), match.end(), index) for match in re.finditer(pattern, text)
            )
    # 同系列长型号优先，防止 Pro Lite 被 Pro 的前缀抢占；同标识冲突行均保留。
    return sorted(
        item
        for item in matches
        if not any(other[0] <= item[0] and other[1] > item[1] for other in matches)
    )


def _unknown_codes(text: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    """提问和回答中的型号标识都必须有真值，未知型号不能借默认单行通过。"""
    references = _mentions(text, rows)
    patterns = (
        r"(?<![a-z0-9_-])(?:[a-z]{1,12}(?:[-_][a-z0-9]+)+|"
        r"[a-z]{1,12}\d[a-z0-9_-]*)(?![a-z0-9_-])",
        r"(?:sku|商品编码|商品编号|型号)\s*[:：=]?\s+([a-z0-9_-]+)",
        r"(?:sku|商品编码|商品编号|型号)\s*[:：=]\s*([a-z0-9_-]+)",
    )
    result = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            group = 1 if match.lastindex else 0
            if not any(
                start <= match.start(group) and end >= match.end(group)
                for start, end, _ in references
            ):
                result.append(match[group])
    return result


def _selected_attributes(context: str, keys: Sequence[str]) -> list[str]:
    matches = [
        (match.start(), match.end(), key)
        for key in keys
        for alias in _aliases(key)
        if alias
        for match in re.finditer(re.escape(alias), context)
    ]
    # 先去掉被更具体属性包含的泛化属性，再按最近出现的字段归属。
    matches = [
        item
        for item in matches
        if not any(
            other[0] <= item[0] and other[1] >= item[1] and other[:2] != item[:2]
            for other in matches
        )
    ]
    if not matches:
        return []
    latest = max(item[0] for item in matches)
    return list(dict.fromkeys(key for start, _, key in matches if start == latest))


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    corrections: str
    checked_count: int


class SKUVerifier:
    """无法唯一定位商品或属性时拒绝生成参数；调用方决定重生成和工单降级。"""

    def verify(
        self,
        answer: str,
        rows: Sequence[dict[str, Any]],
        *,
        query: str = "",
        evidence_texts: Sequence[str] = (),
    ) -> VerificationResult:
        normalized = _normalize(answer)
        row_list = list(rows)
        identifiers = _mentions(normalized, row_list)
        claims = [
            claim
            for claim in extract_quantities(answer)
            if not any(start <= claim.start and end >= claim.end for start, end, _ in identifiers)
        ]
        query_text = _normalize(query)
        query_rows = {index for _, _, index in _mentions(query_text, row_list)}
        if claims and not identifiers and not query_rows and evidence_texts:
            # SOP 中的纸宽等参数可以原样引用授权证据，不能误当成未知商品规格。
            # 只接受含足够上下文的完整原句；孤立数值或改写数值仍走严格核验。
            def compact(text):
                return re.sub(r"\s+", "", _normalize(text))

            evidence = [compact(text) for text in evidence_texts]
            grounded = True
            for claim in claims:
                start = max(
                    (normalized.rfind(mark, 0, claim.start) for mark in "。；;\n"), default=-1
                )
                ends = [normalized.find(mark, claim.end) for mark in "。；;\n"]
                end = min((pos for pos in ends if pos >= 0), default=len(normalized))
                clause = compact(normalized[start + 1 : end])
                grounded &= len(clause) >= len(compact(claim.text)) + 6 and any(
                    clause in source for source in evidence
                )
            suspicious = list(_SUSPICIOUS.finditer(normalized))
            if grounded and all(
                any(c.start <= m.start() and c.end >= m.end() for c in claims) for m in suspicious
            ):
                return VerificationResult(True, "", len(claims))
        errors: list[str] = []
        # 已知参数语境中未支持的单位也不能因为正则未提取就被放行。
        known_keys = [str(key) for row in rows if _identifiers(row) for key in row]
        known_keys.extend(group[0] for group in _ATTRIBUTE_GROUPS)
        for unknown in _SUSPICIOUS.finditer(normalized):
            if not any(
                claim.start <= unknown.start() and claim.end >= unknown.end() for claim in claims
            ) and not any(
                start <= unknown.start() and end >= unknown.end() for start, end, _ in identifiers
            ):
                errors.append(f"无法确定单位量纲：{unknown.group()}，请按商品行原单位重写")
        for key in known_keys:
            for alias in _aliases(key):
                if not alias:
                    continue
                bare_pattern = re.escape(alias) + rf"\s*(?:为|是|[:：=])?\s*(?P<number>{_NUMBER})"
                for bare in re.finditer(bare_pattern, normalized):
                    if not any(claim.start <= bare.start("number") < claim.end for claim in claims):
                        errors.append(f"属性 {alias} 的数值缺少可验证单位，请使用真值单位")
        if claims:
            for source, source_text in (("回答", normalized), ("提问", query_text)):
                for code in _unknown_codes(source_text, row_list):
                    errors.append(f"{source}商品 {code} 不在召回真值中，请明确使用已核验 SKU")
        # 不识别的自然语言数词不能被当成“无参数”答复放行。
        for unsupported in re.finditer(
            rf"[零〇一二两三四五六七八九十百千万亿点半]+\s*{_UNIT}", normalized
        ):
            errors.append(
                f"参数 {unsupported.group()} 使用未支持的数词，请使用阿拉伯数字和真值单位"
            )
        for claim in claims:
            previous_boundary = max(
                (normalized.rfind(symbol, 0, claim.start) for symbol in "\n。；;!?！？"), default=-1
            )
            next_boundaries = [normalized.find(symbol, claim.end) for symbol in "\n。；;!?！？"]
            next_boundary = min(
                (index for index in next_boundaries if index >= 0), default=len(normalized)
            )
            clause = normalized[previous_boundary + 1 : next_boundary]
            local_start = claim.start - previous_boundary - 1
            references = _mentions(clause, row_list)
            previous_claim_end = max(
                (
                    other.end - previous_boundary - 1
                    for other in claims
                    if previous_boundary < other.end <= claim.start
                ),
                default=0,
            )
            new_references = {
                index
                for start, end, index in references
                if previous_claim_end <= start and end <= local_start
            }
            prior_references = [(end, index) for _, end, index in references if end <= local_start]
            if new_references:
                candidates = new_references
            elif prior_references:
                last_end = max(end for end, _ in prior_references)
                candidates = {index for end, index in prior_references if end == last_end}
            elif len({index for _, _, index in references}) == 1:
                candidates = {index for _, _, index in references}
            else:
                prior = [(end, index) for _, end, index in identifiers if end <= claim.start]
                if prior:
                    latest = max(end for end, _ in prior)
                    candidates = {index for end, index in prior if end == latest}
                else:
                    candidates = query_rows or ({0} if len(row_list) == 1 else set())
            if not candidates:
                errors.append(f"参数 {claim.text.strip()} 无法唯一定位 SKU，请明确商品编码")
                continue
            if query_rows and not candidates <= query_rows:
                errors.append(f"参数 {claim.text.strip()} 所属商品与提问 SKU 不一致")
                continue
            context = clause[previous_claim_end:local_start]
            if re.search(
                r"(?:不是|并非|不等于|超过|大于|小于|高于|低于|至少|至多|[<>≤≥≠])\s*$", context
            ):
                errors.append(
                    f"参数 {claim.text.strip()} 带有未验证的否定或比较关系，请按真值直接陈述"
                )
                continue
            for index in candidates:
                row = row_list[index]
                keys = [str(key) for key in row]
                attributes = _selected_attributes(context, keys)
                explicit = _selected_attributes(context, known_keys)
                if (
                    not attributes
                    and explicit
                    and any(name not in {"重量", "功率", "温度"} for name in explicit)
                ):
                    errors.append(
                        f"当前 SKU 没有属性 {'、'.join(explicit)} 的真值，不能从其他属性推断"
                    )
                    continue
                if not attributes:
                    attributes = _selected_attributes(query_text, keys)
                fields = {
                    str(key): (
                        []
                        if _normalize(key).replace(" ", "") in _ID_KEYS
                        else _cell_quantities(str(key), value)
                    )
                    for key, value in row.items()
                }
                if not attributes:
                    attributes = [
                        key
                        for key, values in fields.items()
                        if any(value.dimension == claim.dimension for value in values)
                    ]
                identity = next(iter(_identifiers(row)), "当前商品")
                if len(attributes) != 1:
                    errors.append(
                        f"{identity} 的 {claim.text.strip()} 无法唯一定位属性，"
                        "请标明净重、毛重等字段"
                    )
                    continue
                attribute = attributes[0]
                truths = fields[attribute]
                if not any(self._equal(claim, truth, context) for truth in truths):
                    errors.append(
                        f"{identity} 的 {attribute} 真值为 {_cell_text(row[attribute])}；"
                        f"回答为 {claim.text.strip()}，数值、单位或区间不一致"
                    )
        if errors:
            instruction = (
                "请仅依据下列商品行真值纠正回答，写明 SKU 与参数字段，保留量纲和区间，"
                "不要引用错误参数，也不要补充未证实的参数：\n" + "\n".join(dict.fromkeys(errors))
            )
            return VerificationResult(False, instruction, len(claims))
        return VerificationResult(True, "", len(claims))

    @staticmethod
    def _equal(claim: Quantity, truth: Quantity, context: str) -> bool:
        if claim.dimension != truth.dimension or claim.dimension == "invalid":
            return False
        if claim.kind == truth.kind and claim.values == truth.values:
            return True
        if claim.kind == "scalar" and truth.kind == "range":
            if any(word in context for word in ("上限", "最高")):
                return claim.values == (truth.values[-1],)
            if any(word in context for word in ("下限", "最低")):
                return claim.values == (truth.values[0],)
        return False


async def load_sku_rows(
    session_factory: async_sessionmaker[AsyncSession], parent_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """仅回读已经过检索 ACL 的父块；不按文档扩展扫描、不读其他商品上下文。"""
    if not parent_ids:
        return []
    async with session_factory() as session:
        payloads = await session.scalars(
            select(KBParentChunk.sku_rows_json).where(KBParentChunk.parent_id.in_(parent_ids))
        )
        rows = []
        seen: set[str] = set()
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get("rows"), list):
                candidates = payload["rows"]
            else:
                # 兼容 M1 中以 SKU 为顶层键的结构化参数。
                candidates = [
                    dict(value, SKU=key)
                    for key, value in payload.items()
                    if isinstance(value, dict)
                ]
            for row in candidates:
                if not isinstance(row, dict):
                    continue
                signature = json.dumps(row, ensure_ascii=False, sort_keys=True)
                if signature not in seen:
                    seen.add(signature)
                    rows.append(row)
    return rows
