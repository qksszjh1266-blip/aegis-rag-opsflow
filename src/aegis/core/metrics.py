"""进程级 Prometheus 指标；标签仅使用有限集合及配置中的服务名。"""

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "aegis_http_requests_total", "HTTP 请求总数", ["method", "route", "status_code"]
)
HTTP_REQUEST_DURATION = Histogram(
    "aegis_http_request_duration_seconds",
    "HTTP 请求处理耗时（秒）",
    ["method", "route", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
LOCK_CONFLICTS = Counter("aegis_redis_lock_conflicts_total", "Redis 会话锁冲突总数")
CIRCUIT_STATE = Gauge(
    "aegis_circuit_breaker_state", "熔断器状态：关闭=0，开启=1，半开=2", ["service"]
)
CIRCUIT_REJECTIONS = Counter(
    "aegis_circuit_breaker_rejections_total", "熔断器拦截请求总数", ["service"]
)

LLM_TOKENS = Counter(
    "aegis_llm_tokens_total", "模型输入与输出 Token 用量", ["provider", "direction", "purpose"]
)
