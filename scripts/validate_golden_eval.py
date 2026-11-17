"""校验黄金评测集与 ACL 红线集：语法、字段、doc_code 一致性、ACL 语义与红线词覆盖。

doc_code 与权限唯一事实源为 scripts/init_kb.py 的 SEED_MANIFEST；高危词判定复用
HighRiskPrecheck（含全角与格式控制符归一化），标题关键短语与泄露词必须是种子文档原文子串。
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from aegis.db.models import DocumentType
from aegis.guardrails.precheck import HighRiskPrecheck

ROOT = Path(__file__).resolve().parents[1]

GOLDEN_CATEGORIES = {
    "SKU_SPEC",
    "WAREHOUSE_SOP",
    "AFTER_SALES",
    "FINANCE_AUTHORIZED",
    "HIGH_RISK_P0",
    "NUMERIC_EQUIVALENCE",
}
REDTEAM_ATTACKS = {"LOW_ROLE_ESCALATION", "CROSS_DEPT_PROBE", "PROMPT_INJECTION_SPOOF"}
# 与 ControlledWorkflow 实际输出的 guardrail_action 词汇一致。
WORKFLOW_ACTIONS = {
    "PRECHECK_SHORT_CIRCUIT",
    "SLOT_CLARIFICATION",
    "SLOT_LIMIT_HANDOFF",
    "NO_EVIDENCE",
    "SKU_VERIFIED_PASS",
    "SKU_REGENERATED_PASS",
    "SKU_HALLUCINATION_BLOCKED",
    "STEP_LIMIT_HANDOFF",
    "WORKFLOW_TIMEOUT_HANDOFF",
    "DEPENDENCY_FAILURE_HANDOFF",
}
GOLDEN_TERMINAL_ACTIONS = {"SKU_VERIFIED_PASS", "PRECHECK_SHORT_CIRCUIT"}
NUMERIC_CATEGORIES = {"SKU_SPEC", "AFTER_SALES", "FINANCE_AUTHORIZED", "NUMERIC_EQUIVALENCE"}

GOLDEN_FIELDS = (
    "case_id",
    "category",
    "query",
    "user_context",
    "expected_doc_codes",
    "expected_header_keywords",
    "reference_answer",
    "expected_numbers",
    "expected_guardrail_action",
)
REDTEAM_FIELDS = (
    "case_id",
    "attack_type",
    "query",
    "user_context",
    "forbidden_doc_codes",
    "forbidden_leakage_terms",
    "acceptable_guardrail_actions",
)


def load_manifest() -> dict[str, tuple[str, DocumentType, str, int]]:
    """从 init_kb.py 导入 SEED_MANIFEST，避免评测集与导入清单各写一份权限。"""
    path = ROOT / "scripts" / "init_kb.py"
    spec = importlib.util.spec_from_file_location("init_kb_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        code: (filename, doc_type, dept, level)
        for filename, code, doc_type, dept, level in module.SEED_MANIFEST
    }


def load_dataset(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: JSON 语法或读取失败：{exc}")
        return None
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        errors.append(f"{path.name}: 顶层必须为含 cases 列表的对象")
        return None
    for field in ("dataset_name", "version", "description", "conventions"):
        if field not in data:
            errors.append(f"{path.name}: 缺少元数据字段 {field}")
    return data


def check_context(ctx: Any, label: str, errors: list[str]) -> tuple[list[str], int] | None:
    if not isinstance(ctx, dict) or set(ctx) != {"dept_scope", "role_level"}:
        errors.append(f"{label}: user_context 必须仅含 dept_scope 与 role_level")
        return None
    depts, role = ctx["dept_scope"], ctx["role_level"]
    if (
        not isinstance(depts, list)
        or not depts
        or not all(isinstance(d, str) and d.strip() for d in depts)
    ):
        errors.append(f"{label}: dept_scope 必须为非空字符串列表")
        return None
    if type(role) is not int or role < 0:
        errors.append(f"{label}: role_level 必须为非负整数")
        return None
    return depts, role


def check_str_list(value: Any, label: str, field: str, errors: list[str], *, empty: bool) -> bool:
    ok = isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)
    if ok and not empty:
        ok = bool(value)
    if not ok:
        errors.append(f"{label}: {field} 必须为{'非空' if not empty else ''}字符串列表")
    return ok


def acl_allows(depts: list[str], role: int, dept: str, min_level: int) -> bool:
    return dept in depts and role >= min_level


def validate_golden(
    data: dict[str, Any],
    manifest: dict[str, tuple[str, DocumentType, str, int]],
    texts: dict[str, str],
    precheck: HighRiskPrecheck,
    errors: list[str],
) -> Counter:
    seen: set[str] = set()
    stats: Counter = Counter()
    for case in data["cases"]:
        cid = case.get("case_id", "<缺少 case_id>")
        label = f"golden/{cid}"
        if not isinstance(case, dict):
            errors.append(f"{label}: 样本必须为对象")
            continue
        for field in GOLDEN_FIELDS:
            if field not in case:
                errors.append(f"{label}: 缺少字段 {field}")
        if cid in seen:
            errors.append(f"{label}: case_id 重复")
        seen.add(cid)
        category = case.get("category")
        if category not in GOLDEN_CATEGORIES:
            errors.append(f"{label}: category 非法：{category}")
        for field in ("query", "reference_answer"):
            if not isinstance(case.get(field), str) or not case.get(field, "").strip():
                errors.append(f"{label}: {field} 必须为非空字符串")
        ctx = check_context(case.get("user_context"), label, errors)
        codes = case.get("expected_doc_codes")
        keywords = case.get("expected_header_keywords")
        numbers = case.get("expected_numbers")
        action = case.get("expected_guardrail_action")
        if action not in GOLDEN_TERMINAL_ACTIONS:
            errors.append(f"{label}: expected_guardrail_action 非法：{action}")
        check_str_list(codes, label, "expected_doc_codes", errors, empty=False)
        check_str_list(keywords, label, "expected_header_keywords", errors, empty=False)
        check_str_list(numbers, label, "expected_numbers", errors, empty=True)
        if category in NUMERIC_CATEGORIES and isinstance(numbers, list) and not numbers:
            errors.append(f"{label}: 数值类样本 expected_numbers 不能为空")
        if not isinstance(codes, list) or ctx is None:
            continue
        depts, role = ctx
        joined = ""
        for code in codes:
            if code not in manifest:
                errors.append(f"{label}: doc_code {code} 不在 SEED_MANIFEST")
                continue
            _, _, dept, min_level = manifest[code]
            if not acl_allows(depts, role, dept, min_level):
                errors.append(f"{label}: user_context 不满足 {code} 的 ACL（{dept}/{min_level}）")
            joined += texts[code]
        if isinstance(keywords, list):
            for keyword in keywords:
                if isinstance(keyword, str) and keyword not in joined:
                    errors.append(f"{label}: 标题关键短语不在期望文档原文中：{keyword}")
        risks = precheck.check(case.get("query", ""))
        if category == "HIGH_RISK_P0":
            if not risks:
                errors.append(f"{label}: P0 样本 query 未命中任何强制高危词")
            if action != "PRECHECK_SHORT_CIRCUIT":
                errors.append(f"{label}: P0 样本必须期望 PRECHECK_SHORT_CIRCUIT")
        else:
            if risks:
                errors.append(f"{label}: 非 P0 样本 query 意外命中高危词 {risks}")
            if action != "SKU_VERIFIED_PASS":
                errors.append(f"{label}: 常规样本必须期望 SKU_VERIFIED_PASS")
        stats[category] += 1
        stats[f"action:{action}"] += 1
        stats[f"dept:{'+'.join(depts)}/L{role}"] += 1
    return stats


def validate_redteam(
    data: dict[str, Any],
    manifest: dict[str, tuple[str, DocumentType, str, int]],
    texts: dict[str, str],
    precheck: HighRiskPrecheck,
    errors: list[str],
) -> Counter:
    seen: set[str] = set()
    stats: Counter = Counter()
    for case in data["cases"]:
        cid = case.get("case_id", "<缺少 case_id>")
        label = f"redteam/{cid}"
        if not isinstance(case, dict):
            errors.append(f"{label}: 样本必须为对象")
            continue
        for field in REDTEAM_FIELDS:
            if field not in case:
                errors.append(f"{label}: 缺少字段 {field}")
        if cid in seen:
            errors.append(f"{label}: case_id 重复")
        seen.add(cid)
        attack = case.get("attack_type")
        if attack not in REDTEAM_ATTACKS:
            errors.append(f"{label}: attack_type 非法：{attack}")
        if not isinstance(case.get("query"), str) or not case.get("query", "").strip():
            errors.append(f"{label}: query 必须为非空字符串")
        ctx = check_context(case.get("user_context"), label, errors)
        codes = case.get("forbidden_doc_codes")
        terms = case.get("forbidden_leakage_terms")
        actions = case.get("acceptable_guardrail_actions")
        check_str_list(codes, label, "forbidden_doc_codes", errors, empty=False)
        check_str_list(terms, label, "forbidden_leakage_terms", errors, empty=False)
        if not isinstance(actions, list) or not actions or not set(actions) <= WORKFLOW_ACTIONS:
            errors.append(f"{label}: acceptable_guardrail_actions 必须为工作流动作词汇的非空子集")
        if precheck.check(case.get("query", "")):
            errors.append(f"{label}: 刺探 query 含高危词，会被前置短路，无法检验 ACL")
        if not isinstance(codes, list) or ctx is None:
            continue
        depts, role = ctx
        joined = ""
        for code in codes:
            if code not in manifest:
                errors.append(f"{label}: doc_code {code} 不在 SEED_MANIFEST")
                continue
            _, _, dept, min_level = manifest[code]
            if acl_allows(depts, role, dept, min_level):
                errors.append(f"{label}: user_context 竟满足 {code} 的 ACL，刺探不成立")
            joined += texts[code]
        if isinstance(terms, list):
            for term in terms:
                if isinstance(term, str) and term not in joined:
                    errors.append(f"{label}: 泄露词不在 forbidden 文档原文中：{term}")
        stats[attack] += 1
        stats[f"dept:{'+'.join(depts)}/L{role}"] += 1
    return stats


def split_stats(stats: Counter, prefix: str) -> dict[str, int]:
    return {k[len(prefix) :]: v for k, v in stats.items() if k.startswith(prefix)}


def main() -> int:
    cli = argparse.ArgumentParser(description="校验黄金评测集与 ACL 红线集的结构、一致性与红线覆盖")
    cli.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "data" / "golden_eval" / "golden_qa_dataset.json",
    )
    cli.add_argument(
        "--redteam",
        type=Path,
        default=ROOT / "data" / "golden_eval" / "acl_redteam_dataset.json",
    )
    args = cli.parse_args()

    errors: list[str] = []
    manifest = load_manifest()
    texts = {
        code: (ROOT / "data" / "seed_docs" / filename).read_text(encoding="utf-8")
        for code, (filename, _, _, _) in manifest.items()
    }
    precheck = HighRiskPrecheck()
    golden = load_dataset(args.golden, errors)
    redteam = load_dataset(args.redteam, errors)
    golden_stats: Counter = Counter()
    redteam_stats: Counter = Counter()
    if golden is not None:
        if len(golden["cases"]) < 24:
            errors.append(f"golden: 样本数 {len(golden['cases'])} 少于 24")
        golden_stats = validate_golden(golden, manifest, texts, precheck, errors)
    if redteam is not None:
        if len(redteam["cases"]) < 10:
            errors.append(f"redteam: 样本数 {len(redteam['cases'])} 少于 10")
        redteam_stats = validate_redteam(redteam, manifest, texts, precheck, errors)

    print("=== 黄金基准评测集分布 ===")
    for key in sorted(k for k in golden_stats if not k.startswith(("action:", "dept:"))):
        print(f"  {key}: {golden_stats[key]}")
    print("  护栏动作:", split_stats(golden_stats, "action:"))
    print("  权限上下文:", split_stats(golden_stats, "dept:"))
    print("=== ACL 红线集分布 ===")
    for key in sorted(k for k in redteam_stats if not k.startswith("dept:")):
        print(f"  {key}: {redteam_stats[key]}")
    print("  权限上下文:", split_stats(redteam_stats, "dept:"))
    covered = {
        c
        for case in (golden["cases"] if golden else [])
        for c in case.get("expected_doc_codes", [])
    }
    targeted = {
        c
        for case in (redteam["cases"] if redteam else [])
        for c in case.get("forbidden_doc_codes", [])
    }
    print("=== 文档覆盖 ===")
    for code in manifest:
        marks = []
        if code in covered:
            marks.append("黄金集覆盖")
        if code in targeted:
            marks.append("红线集目标")
        print(f"  {code}: {'、'.join(marks) or '未覆盖'}")
    if errors:
        print(f"\n校验失败，共 {len(errors)} 项：")
        for error in errors:
            print(f"  - {error}")
        return 1
    golden_n = len(golden["cases"]) if golden else 0
    redteam_n = len(redteam["cases"]) if redteam else 0
    print(f"\n校验通过：黄金集 {golden_n} 条 + 红线集 {redteam_n} 条 = {golden_n + redteam_n} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
