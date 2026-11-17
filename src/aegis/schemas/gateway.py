"""REST 与 MCP 共用的业务输入、结果和量纲明细契约。"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from aegis.db.models import DocumentType


class GatewayRequest(BaseModel):
    """禁止夹带权限字段；权限始终由认证上下文确定。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchRequest(GatewayRequest):
    query: str = Field(min_length=1, max_length=16000, description="待检索的企业知识问题")
    top_k: int = Field(default=5, ge=1, le=30, strict=True, description="返回父块数量上限")
    doc_type: DocumentType | None = Field(default=None, description="可选文档类型过滤条件")


class SearchHit(BaseModel):
    parent_id: str = Field(description="可供规格核验接口回读的父块标识")
    doc_code: str = Field(description="知识文档唯一编码")
    content_text: str = Field(description="经过权限过滤的完整父块内容")
    header_path: str = Field(description="来源文档的标题层级路径")
    rrf_score: float = Field(description="双路召回融合分数")
    rerank_score: float | None = Field(default=None, description="精排分数，降级时为空")


class SearchResponse(BaseModel):
    hits: list[SearchHit] = Field(description="授权范围内的父块召回结果")
    rerank_fallback: bool = Field(description="是否发生精排失败并回退融合排序")
    latency_ms: float = Field(ge=0, description="服务检索耗时，单位毫秒")
    trace_id: str = Field(description="全链路追踪标识")


class VerifySKURequest(GatewayRequest):
    candidate_text: str = Field(min_length=1, max_length=16000, description="待核验的候选回复")
    sku: str = Field(min_length=1, max_length=128, description="需要核验的准确商品编码")
    parent_id: str = Field(min_length=1, max_length=64, description="检索返回的规格父块标识")


class NormalizedQuantity(BaseModel):
    text: str = Field(description="原始参数文本")
    dimension: str = Field(description="规范化量纲，如质量、体积、功率对应的内部类型")
    values: list[str] = Field(description="规范化十进制数值，质量用克、体积用毫升、长度用毫米")
    kind: str = Field(description="标量、区间、公差或多维尺寸类型")


class ReferenceQuantity(NormalizedQuantity):
    attribute: str = Field(description="真值所属的商品属性名称")


class VerifySKUResponse(BaseModel):
    passed: bool = Field(description="候选回复是否全部通过行级量纲校验")
    corrections: str = Field(description="不一致时返回依据商品真值生成的纠正提示")
    checked_count: int = Field(ge=0, description="已识别并核验的参数数量")
    claims: list[NormalizedQuantity] = Field(description="候选回复中提取的规范化参数")
    reference_values: list[ReferenceQuantity] = Field(description="目标商品行的规范化参考真值")
    trace_id: str = Field(description="全链路追踪标识")


class CreateTicketRequest(GatewayRequest):
    idempotency_key: str = Field(min_length=1, max_length=128, description="调用方生成的防重键")
    session_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$", description="业务会话标识"
    )
    message: str = Field(min_length=1, max_length=16000, description="需要履约处理的问题描述")
    slots_payload: dict[
        Annotated[str, Field(min_length=1, max_length=64)],
        Annotated[str, Field(max_length=512)],
    ] = Field(
        default_factory=dict,
        max_length=32,
        description="设备编号、订单号或波次号、报错现象等已知槽位，最多三十二项",
    )
