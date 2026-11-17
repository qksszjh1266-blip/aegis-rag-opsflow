"""报告格式、双账本隔离及原子写入失败恢复的回归测试。"""

import json
from pathlib import Path

import pytest

from aegis.eval.report import render_console, render_markdown, write_reports


def report():
    return {
        "run_id": "运行甲",
        "generated_at": "2026-11-17T08:00:00Z",
        "config": {"concurrency": 5, "top_k": 3, "app_provider": "mock", "judge_provider": "mock"},
        "dataset_counts": {"golden": 2, "redteam": 1, "high_risk": 1, "retrieval": 1},
        "metrics": {"hit_at_k": 1.0, "high_risk_intercept_rate": 1.0, "acl_leakage_rate": 0.0},
        "attribution": {"LOW_RANKING": {"count": 1, "ratio": 1 / 3}},
        "app_tokens": {"prompt_tokens": 11, "completion_tokens": 12, "total_tokens": 23},
        "judge_tokens": {"prompt_tokens": 31, "completion_tokens": 32, "total_tokens": 63},
        "gate": {"passed": False, "reasons": ["高危样本的持久化工单证据不完整"]},
        "cases": [
            {
                "case_id": "失败样本",
                "suite": "golden",
                "category": "SKU_SPEC",
                "passed": False,
                "answer": "<script>伪造</script>|参数\n错误",
                "verdict": "上下文缺失",
                "badcase_category": "LOW_RANKING",
                "raw_parent_ids": ["真实粗召回父块"],
                "final_parent_ids": ["最终父块"],
                "latency_ms": 1.5,
            }
        ],
    }


def test_report_keeps_ledgers_separate_and_escapes_model_text(tmp_path):
    value = report()
    json_path, markdown_path = write_reports(value, tmp_path)
    assert json_path.name == "latest_eval_report.json"
    assert markdown_path.name == "latest_eval_report.md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["app_tokens"]["total_tokens"] == 23
    assert saved["judge_tokens"]["total_tokens"] == 63
    text = markdown_path.read_text(encoding="utf-8")
    assert "| 业务应用 AppTokenLedger | 11 | 12 | 23 |" in text
    assert "| 语义裁判 JudgeTokenLedger | 31 | 32 | 63 |" in text
    assert "| ACL 越权泄漏率 | 0.00% |" in text
    assert "真实粗召回父块" in text and "最终父块" in text
    assert "<script>" not in text and "&lt;script&gt;" in text and "\\|参数 错误" in text
    assert "高危样本的持久化工单证据不完整" in render_console(value)
    assert "缺失必要样本" in render_markdown(value)


def test_report_invalid_json_does_not_replace_previous_report(tmp_path):
    json_path, markdown_path = write_reports(report(), tmp_path)
    before = (json_path.read_bytes(), markdown_path.read_bytes())
    invalid = report()
    invalid["metrics"]["hit_at_k"] = float("nan")
    with pytest.raises(ValueError):
        write_reports(invalid, tmp_path)
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before


def test_report_replace_failure_preserves_previous_file_and_removes_temporary(
    tmp_path, monkeypatch
):
    json_path, markdown_path = write_reports(report(), tmp_path)
    before = json_path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("模拟磁盘替换失败")

    monkeypatch.setattr("aegis.eval.report.os.replace", fail_replace)
    with pytest.raises(OSError, match="模拟磁盘替换失败"):
        write_reports(report(), tmp_path)
    assert json_path.read_bytes() == before
    assert set(Path(tmp_path).iterdir()) == {json_path, markdown_path}
