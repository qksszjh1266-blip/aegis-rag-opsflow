"""评测命令行的红线退出码、资源回收与参数校验。"""

import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aegis.config import Settings

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "eval_command_test", _ROOT / "scripts" / "run_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harness(module, monkeypatch, tmp_path, *, passed, enforce):
    """仅替换模型与存储边界，保留命令行评测控制流及门禁退出码逻辑。"""
    settings = Settings(_env_file=None)
    lifecycle = []

    @asynccontextmanager
    async def lifespan(app):
        lifecycle.append("打开")
        try:
            yield
        finally:
            lifecycle.append("关闭")

    app = SimpleNamespace(
        state=SimpleNamespace(workflow=object(), session_factory=object()),
        router=SimpleNamespace(lifespan_context=lifespan),
    )
    result = {"gate": {"passed": passed, "reasons": [] if passed else ["ACL 泄漏"]}}
    runner = SimpleNamespace(run=AsyncMock(return_value=result))
    judge = SimpleNamespace(aclose=AsyncMock())
    initialize = AsyncMock()
    monkeypatch.setattr(module, "Settings", lambda: settings)
    monkeypatch.setattr(module, "load_datasets", lambda *args: (["黄金样本"], ["红队样本"]))
    monkeypatch.setattr(module, "initialize", initialize)
    monkeypatch.setattr(module, "create_app", lambda _: app)
    monkeypatch.setattr(module, "AnswerJudge", lambda *args, **kwargs: judge)
    monkeypatch.setattr(module, "EvaluationRunner", lambda *args, **kwargs: runner)
    writer = Mock(return_value=(tmp_path / "报告.json", tmp_path / "报告.md"))
    monkeypatch.setattr(module, "write_reports", writer)
    args = SimpleNamespace(
        golden=tmp_path / "golden.json",
        redteam=tmp_path / "redteam.json",
        concurrency=5,
        top_k=5,
        min_answer_pass_rate=0.9,
        output_dir=tmp_path,
        enforce_gate=enforce,
        judge_provider="mock",
    )
    return args, runner, judge, initialize, lifecycle, writer


@pytest.mark.parametrize(
    "passed,enforce,expected",
    [(True, True, 0), (False, True, 1), (False, False, 0), (True, False, 0)],
)
async def test_eval_cli_gate_exit_code(script, monkeypatch, tmp_path, passed, enforce, expected):
    args, runner, judge, initialize, lifecycle, writer = harness(
        script, monkeypatch, tmp_path, passed=passed, enforce=enforce
    )
    assert await script.evaluate(args) == expected
    initialize.assert_awaited_once()
    runner.run.assert_awaited_once_with(["黄金样本"], ["红队样本"])
    writer.assert_called_once()
    judge.aclose.assert_awaited_once()
    assert lifecycle == ["打开", "关闭"]


async def test_eval_cli_report_failure_still_releases_resources(script, monkeypatch, tmp_path):
    args, _, judge, _, lifecycle, writer = harness(
        script, monkeypatch, tmp_path, passed=True, enforce=True
    )
    writer.side_effect = OSError("报告目录不可写")
    with pytest.raises(OSError, match="报告目录不可写"):
        await script.evaluate(args)
    judge.aclose.assert_awaited_once()
    assert lifecycle == ["打开", "关闭"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--concurrency", "0"],
        ["--top-k", "31"],
        ["--min-answer-pass-rate", "1.1"],
        ["--min-answer-pass-rate", "nan"],
    ],
)
def test_eval_cli_invalid_limits_fail_before_startup(script, monkeypatch, arguments):
    evaluate = AsyncMock()
    monkeypatch.setattr(script, "evaluate", evaluate)
    monkeypatch.setattr("sys.argv", ["run_eval.py", *arguments])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    evaluate.assert_not_awaited()


def test_eval_cli_main_propagates_gate_exit_code(script, monkeypatch):
    evaluate = AsyncMock(return_value=1)
    monkeypatch.setattr(script, "evaluate", evaluate)
    monkeypatch.setattr("sys.argv", ["run_eval.py", "--enforce-gate", "--concurrency", "3"])
    assert script.main() == 1
    args = evaluate.call_args.args[0]
    assert args.enforce_gate and args.concurrency == 3
