"""黄金集与红队集的严格输入契约，权限与期望值绝不传给业务模型。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationContext(BaseModel):
    """只定义测试调用方的真实权限，主体标识由运行器独立生成。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    dept_scope: list[str] = Field(min_length=1)
    role_level: int = Field(ge=0, strict=True)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    case_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=16000)
    user_context: EvaluationContext


class GoldenCase(EvaluationCase):
    category: Literal[
        "SKU_SPEC",
        "WAREHOUSE_SOP",
        "AFTER_SALES",
        "FINANCE_AUTHORIZED",
        "HIGH_RISK_P0",
        "NUMERIC_EQUIVALENCE",
    ]
    expected_doc_codes: list[str] = Field(min_length=1)
    expected_header_keywords: list[str] = Field(default_factory=list)
    expected_parent_ids: list[str] = Field(default_factory=list)
    reference_answer: str = Field(min_length=1)
    expected_numbers: list[str]
    expected_guardrail_action: str


class RedteamCase(EvaluationCase):
    attack_type: Literal["LOW_ROLE_ESCALATION", "CROSS_DEPT_PROBE", "PROMPT_INJECTION_SPOOF"]
    forbidden_doc_codes: list[str] = Field(min_length=1)
    forbidden_leakage_terms: list[str] = Field(min_length=1)
    acceptable_guardrail_actions: list[str] = Field(min_length=1)
