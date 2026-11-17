"""启动工单消费者，恢复事务发件箱并持续处理派单和死信。"""

import argparse
import asyncio
import json
import signal
from contextlib import AsyncExitStack

from aegis.config import Settings
from aegis.core.logging import configure_logging
from aegis.db.redis_client import create_redis_client
from aegis.db.session import create_mysql_engine, create_session_factory
from aegis.workers.consumer import TicketWorker


async def run(*, once: bool, consumer: str | None, poll_interval: float) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    # asyncio.run 已处理 SIGINT；容器停止时的 SIGTERM 同样走协程取消和资源关闭。
    installed = False
    if task is not None:
        try:
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            installed = True
        except NotImplementedError:
            pass
    try:
        async with AsyncExitStack() as stack:
            engine = create_mysql_engine(settings)
            stack.push_async_callback(engine.dispose)
            redis = create_redis_client(settings)
            stack.push_async_callback(redis.aclose)
            worker = TicketWorker(redis, create_session_factory(engine), consumer_name=consumer)
            if once:
                processed = await worker.run_once()
                print(json.dumps({"processed": processed}, ensure_ascii=False))
            else:
                await worker.run_forever(poll_interval=poll_interval)
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复事务发件箱并消费工单派单事件", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("--once", action="store_true", help="只执行一轮后退出，默认持续运行")
    parser.add_argument("--consumer", help="消费者名称，默认生成独立实例标识")
    parser.add_argument(
        "--poll-interval", type=float, default=0.5, help="持续运行的轮询间隔，单位秒"
    )
    args = parser.parse_args()
    if not 0 < args.poll_interval <= 60:
        parser.error("轮询间隔必须在 0 至 60 秒内")
    try:
        asyncio.run(run(once=args.once, consumer=args.consumer, poll_interval=args.poll_interval))
    except (KeyboardInterrupt, asyncio.CancelledError):
        # 处理中的消息留在 PEL，后续消费者按空闲时间恢复，不把退出误计为派单失败。
        pass


if __name__ == "__main__":
    main()
