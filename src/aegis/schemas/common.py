"""全局错误与健康探针的中文响应契约。"""

from typing import Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """所有异常响应都保留追踪标识，避免暴露底层连接和凭据信息。"""

    trace_id: str | None = Field(description="全链路追踪标识")
    error_code: str = Field(description="稳定的业务或协议错误编码")
    message: str = Field(description="可向调用方展示的中文错误说明")


ERROR_RESPONSES = {
    code: {"model": ErrorResponse, "description": description}
    for code, description in {
        400: "业务请求不符合处理条件",
        401: "缺少有效的认证凭据",
        404: "资源不存在或当前身份无权访问",
        409: "会话互斥或请求标识发生冲突",
        422: "请求参数不符合接口契约",
        500: "内部处理失败，请提供追踪标识进行排查",
        503: "上游依赖不可用或已触发熔断",
    }.items()
}


class LiveResponse(BaseModel):
    status: Literal["ok"] = Field(description="进程存活状态")
    trace_id: str | None = Field(description="全链路追踪标识")


class DependencyHealth(BaseModel):
    status: Literal["ok", "error"] = Field(description="单项依赖的探测状态")
    error: Literal["timeout", "unavailable"] | None = Field(
        default=None, description="失败时的安全错误类别"
    )
    latency_ms: float = Field(ge=0, description="该依赖探测耗时，单位毫秒")


class ReadyResponse(BaseModel):
    status: Literal["ok", "error"] = Field(description="基础设施整体就绪状态")
    trace_id: str | None = Field(description="全链路追踪标识")
    components: dict[str, DependencyHealth] = Field(description="各基础设施的探测结果")
