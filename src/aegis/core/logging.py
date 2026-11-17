"""统一标准库与 structlog 的 JSON 日志格式。"""

import logging
import sys
from typing import Any

import structlog

from aegis.core.trace_context import get_trace_id


def add_trace_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """强制使用请求上下文中的标识，防止调用方意外覆盖链路字段。"""
    event_dict["trace_id"] = get_trace_id()
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """幂等配置根日志；经过本处理器的输出均包含时间、级别与链路字段。"""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_trace_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    # 仅替换本模块安装的处理器，保留宿主及测试框架的日志接收器。
    for previous in list(root_logger.handlers):
        if getattr(previous, "_aegis_handler", False):
            root_logger.removeHandler(previous)
            previous.close()
    handler._aegis_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
