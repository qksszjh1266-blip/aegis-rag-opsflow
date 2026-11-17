"""受控工作流输入、权限上下文与可持久化返回契约。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AccessContext(BaseModel):
    """由认证适配层提供，禁止从对话正文或模型输出推导权限。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    subject_id: str = Field(min_length=1, max_length=128)
    dept_scope: list[str] = Field(min_length=1)
    role_level: int = Field(ge=0, strict=True)


class AgentRequest(BaseModel):
    """每次会话提交必须指定可重试的请求标识。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$", description="业务会话唯一标识"
    )
    request_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$", description="会话内请求防重标识"
    )
    message: str = Field(min_length=1, max_length=16000, description="当前轮用户问题或补充信息")
    scenario: Literal["qa", "troubleshooting"] = Field(
        default="qa", description="问答或多轮故障排查场景"
    )


class AgentResponse(BaseModel):
    """回答中仅保留通过核验的内容，不返回被拦截的模型原始文本。"""

    answer: str = Field(description="通过护栏的回答或人工处理提示")
    trace_id: str = Field(description="全链路追踪标识")
    guardrail_action: str = Field(description="本轮执行的护栏或降级动作")
    ticket_no: str | None = Field(default=None, description="转人工时创建的履约工单号")
    asked_slots: list[str] = Field(default_factory=list, description="本轮追问的槽位名称")
    completeness_score: float = Field(default=0, ge=0, le=1, description="排障槽位完整度")
    degraded_flag: bool = Field(default=False, description="是否发生检索或工作流降级")
    input_tokens: int = Field(default=0, description="本轮累计模型输入令牌数")
    output_tokens: int = Field(default=0, description="本轮累计模型输出令牌数")
    steps: int = Field(default=0, ge=0, le=8, description="本轮执行的状态机节点数量")
