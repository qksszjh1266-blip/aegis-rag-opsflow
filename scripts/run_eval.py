"""同步种子库后并发执行黄金与红队评测，输出中文报告并按红线阻断发布。"""

import argparse
import asyncio
import math
from pathlib import Path

from init_kb import initialize

from aegis.config import ProviderMode, Settings
from aegis.eval.judge import AnswerJudge
from aegis.eval.report import render_console, write_reports
from aegis.eval.runner import EvaluationRunner, load_datasets
from aegis.main import create_app

ROOT = Path(__file__).resolve().parents[1]


async def evaluate(args) -> int:
    """数据先校验再同步；任何基础设施或配置失败都返回非成功状态。"""
    golden, redteam = load_datasets(args.golden, args.redteam)
    await initialize(ROOT / "data" / "seed_docs")
    settings = Settings()
    app = create_app(settings)
    judge_mode = (
        ProviderMode(args.judge_provider) if args.judge_provider else settings.provider_mode
    )
    judge = AnswerJudge(settings, provider_mode=judge_mode)
    try:
        async with app.router.lifespan_context(app):
            runner = EvaluationRunner(
                app.state.workflow,
                judge,
                app.state.session_factory,
                concurrency=args.concurrency,
                top_k=args.top_k,
                app_provider=settings.llm_provider_mode.value,
                judge_provider=judge_mode.value,
                min_answer_pass_rate=args.min_answer_pass_rate,
            )
            report = await runner.run(golden, redteam)
            json_path, markdown_path = await asyncio.to_thread(
                write_reports, report, args.output_dir
            )
            print(render_console(report))
            print(f"JSON 报告：{json_path}\n中文报告：{markdown_path}")
            return 1 if args.enforce_gate and not report["gate"]["passed"] else 0
    finally:
        await judge.aclose()


def main() -> int:
    cli = argparse.ArgumentParser(description="M5 全量异步评测与红线发布门禁")
    cli.add_argument("--concurrency", type=int, default=5, help="异步并发上限，默认 5")
    cli.add_argument("--top-k", type=int, default=5, help="实际传入业务工作流的上下文父块数")
    cli.add_argument(
        "--min-answer-pass-rate",
        type=float,
        default=0.9,
        help="黄金问答最低通过率，取值 0 至 1，默认 0.9",
    )
    cli.add_argument(
        "--output-dir", type=Path, default=ROOT / "data" / "golden_eval", help="评测产物目录"
    )
    cli.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "data" / "golden_eval" / "golden_qa_dataset.json",
        help="黄金基准集",
    )
    cli.add_argument(
        "--redteam",
        type=Path,
        default=ROOT / "data" / "golden_eval" / "acl_redteam_dataset.json",
        help="ACL 红队集",
    )
    cli.add_argument(
        "--judge-provider",
        choices=[mode.value for mode in ProviderMode],
        help="独立裁判模式，默认继承全局 PROVIDER_MODE",
    )
    cli.add_argument(
        "--enforce-gate", action="store_true", help="红线失败或执行不完整时返回非零退出码"
    )
    args = cli.parse_args()
    if not 1 <= args.concurrency <= 100 or not 1 <= args.top_k <= 30:
        cli.error("并发数必须为 1 至 100，Top-K 必须为 1 至 30")
    if not math.isfinite(args.min_answer_pass_rate) or not 0 <= args.min_answer_pass_rate <= 1:
        cli.error("黄金问答最低通过率必须为 0 至 1 的有限数值")
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
