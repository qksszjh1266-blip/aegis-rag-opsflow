"""可安全返回客户端的业务异常，不包含底层连接或密钥信息。"""


class AegisError(Exception):
    """业务异常基类。"""

    status_code = 400
    error_code = "AEGIS_ERROR"
    default_message = "业务请求处理失败"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class SessionLockBusyError(AegisError):
    """同一会话已有请求正在执行。"""

    status_code = 409
    error_code = "SESSION_LOCK_BUSY"
    default_message = "会话正在处理中，请稍后重试"


class CircuitBreakerOpenError(AegisError):
    """供应商熔断期间拒绝调用。"""

    status_code = 503
    error_code = "CIRCUIT_BREAKER_OPEN"
    default_message = "上游服务暂时不可用，请稍后重试"


class ResourceNotFoundError(AegisError):
    """指定资源不存在。"""

    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"
    default_message = "请求的资源不存在"
