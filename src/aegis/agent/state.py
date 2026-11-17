"""Redis 会话状态与八步预算；状态写入必须同时验证锁所有权。"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from aegis.core.exceptions import AegisError, SessionLockBusyError
from aegis.schemas.agent import AccessContext, AgentResponse


class SessionContextError(AegisError):
    status_code = 409
    error_code = "SESSION_CONTEXT_CONFLICT"
    default_message = "会话权限或请求标识不匹配，请使用新的会话或请求标识"


class StepLimitExceeded(Exception):
    """节点预算耗尽后只能退出循环并转人工。"""


class StepBudget:
    def __init__(self) -> None:
        self.used = 0

    def enter(self) -> None:
        if self.used >= 8:
            raise StepLimitExceeded("单轮工作流最多执行八个节点")
        self.used += 1


class CachedTurn(BaseModel):
    fingerprint: str
    response: AgentResponse


class SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_fingerprint: str
    turn_count: int = Field(default=0, ge=0)
    slots: dict[str, str] = Field(default_factory=dict)
    question_rounds: int = Field(default=0, ge=0, le=2)
    history: list[dict[str, str]] = Field(default_factory=list)
    context_summary: str = ""
    summarized_turns: int = Field(default=0, ge=0)
    completed_requests: dict[str, CachedTurn] = Field(default_factory=dict)
    troubleshooting: bool = False


def access_fingerprint(access: AccessContext) -> str:
    data = access.model_dump()
    data["dept_scope"] = sorted(set(access.dept_scope))
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


_SAVE_STATE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
return 1
"""


class RedisStateStore:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def load(self, session_id: str, access: AccessContext) -> SessionState:
        fingerprint = access_fingerprint(access)
        raw = await self.redis.get(f"fsm:session:{session_id}")
        if raw is None:
            return SessionState(access_fingerprint=fingerprint)
        state = SessionState.model_validate_json(raw)
        if state.access_fingerprint != fingerprint:
            raise SessionContextError()
        return state

    async def save(self, session_id: str, token: str, state: SessionState) -> None:
        saved = await self.redis.eval(
            _SAVE_STATE,
            2,
            f"lock:session:{session_id}",
            f"fsm:session:{session_id}",
            token,
            state.model_dump_json(),
            86400,
        )
        if not saved:
            raise SessionLockBusyError("会话锁已失效，拒绝写入过期状态")
