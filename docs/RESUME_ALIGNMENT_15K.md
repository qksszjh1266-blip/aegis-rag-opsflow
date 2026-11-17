# 15K AI 应用开发岗位：简历与源码逐行对齐终审手册

本手册以 `aegis-rag-opsflow` 当前实现和[最新评测 JSON](../data/golden_eval/latest_eval_report.json)为依据，供简历定稿、现场演示与终面追问使用。星澜智选电商为脱敏示例名称，表示仓库种子知识库所覆盖的业务场景；仓库不能证明候选人的实际任职经历、生产用户规模或商业收益。请按实际身份将下述文案放入「项目经历」，不要自行改成「已在公司全面上线」。

## 一、可直接使用的项目经历

### 星澜智选电商企业知识检索与履约工单智能体

**项目背景：** 围绕星澜智选电商的业务场景，建设可复现的企业知识问答与履约排障系统，覆盖近似型号 SKU 参数咨询、仓配打单扫码 SOP、多平台售后赔付规则及高密财务制度访问隔离。重点解决商品参数混淆、模型编造物理参数、跨部门知识泄漏和重复建单问题。

**技术栈：** Python 3.12+、uv、FastAPI / OpenAPI 3.1、FastMCP SSE、SQLAlchemy 2.0 异步引擎 / asyncmy、MySQL 8.0、Redis 7、Qdrant、千问兼容接口、structlog、Prometheus、pytest。

**核心设计与工程实现：**

- 设计父子块混合检索链路：按 Markdown H1–H3 组织约 2000 / 400 字符父子块，普通文本重叠 50 字符，SKU 表格按完整行切分并保留结构化真值；并发执行 Dense 与 BM25 Sparse 召回，通过父块级 RRF（k=60）融合和精排生成上下文，精排超过 600ms 预算自动回退并落审计日志。
- 将部门与职级权限绑定到服务端认证上下文，在两路 Qdrant 检索前下推 Payload ACL；通过内容与配置指纹识别相同快照，构建新集合后原子切换 Alias，并固定单次检索的物理集合，避免重建期间跨版本拼接。
- 实现 Redis 状态机和确定性护栏：单轮最多 8 个节点、槽位最多追问 2 轮、每 10 轮压缩历史；高危词直接创建 P0 工单，绕过检索与模型；SKU 参数按商品、属性、量纲与数值逐项核验，失败后仅允许 1 次带真值反馈的重生成，持续失败则安全拒答并创建 P1 人工复核工单。
- 建设并发与故障恢复机制：使用随机令牌及 Lua 校验锁所有权，拒绝过期会话状态写入；通过 Redis 滑动窗口三态熔断隔离上游故障；用 24 小时幂等缓存、MySQL 唯一键及事务发件箱保障建单与事件恢复，Redis Streams 消费累计 3 次失败后原子迁入 DLQ。
- 实现 REST / MCP 共用领域服务和异步评测门禁，记录粗召回、最终上下文、生成结果及应用/裁判独立 Token 账本；结合百分比容差判定、SKU 属性绑定和三级瀑布归因定位召回、排序与回答故障，使用真实 MySQL、Redis、Qdrant 集成测试验证关键链路。

**量化验证：** 在本地真实存储、确定性 mock 应用与 mock 裁判模式下，对 40 条黄金问答及 13 条 ACL 红队样本进行并发评测；34 条常规检索样本文档 Hit@5 / Recall@5 为 100.00% / 100.00%，MRR 为 1.0000，父块 Recall@5 为 78.63%；黄金问答通过率 100.00%，6 条高危样本 P0 拦截率 100.00%，13 条红队样本 ACL 泄漏率 0.00%。这些结果表示当前基准集的工程回归表现，不表示真实模型、线上流量或任意攻击下的准确率。

### 指标引用凭证与口径

当前引用运行：`19d42f2c85e041f19205f7a5e12634dd`，生成时间 `2026-11-17T04:22:26.256933+00:00`，并发数 5，Top-K=5，发布门禁通过。原始凭证见[评测 JSON](../data/golden_eval/latest_eval_report.json)和[可读报告](../data/golden_eval/latest_eval_report.md)。

| 指标 | 实测值 | 面试时应主动交代的口径 |
| --- | ---: | --- |
| 文档 Hit@5 / Recall@5 / MRR | 100.00% / 100.00% / 1.0000 | 对 34 条黄金常规样本作宏平均，排除不检索的 P0 分支 |
| 父块 Hit@5 / Recall@5 / MRR | 100.00% / 78.63% / 0.9559 | Recall 分母来自完整授权知识库中符合规则的父块全集 |
| 黄金问答通过率 | 100.00% | 40 条，包含 6 条高危 P0 样本；默认最低门槛为 90% |
| P0 拦截率 | 100.00% | 6 条样本均为单步短路，检索与模型调用均为零 |
| ACL 泄漏率 / 红队动作合规率 | 0.00% / 100.00% | 13 条样本，检查粗召回池、最终上下文、回答和预期护栏动作 |
| 业务端到端 P50 / P95 | 92.6635ms / 157.4898ms | 不含并发排队、裁判和评测归因写回；是 mock 基线，非真实千问时延 |
| 应用输入 / 输出 / 合计 Token | 136515 / 6614 / 143129 | mock 字符估算，不可直接作为 API 账单或成本节省依据 |
| 裁判输入 / 输出 / 合计 Token | 11712 / 1923 / 13635 | 与应用账本隔离；真实供应商模式才使用响应 usage |

## 二、简历参数与源码位置核查表

行号用于定位本次终审版本，后续维护以类/函数名为稳定检索入口。下表所列测试是已经存在的验证入口；完整通过状态应以本次全量回归结果为准。

| 简历中的技术点 | 源码与核心类 / 函数 | 真实行为与表述边界 | 对应验证文件 / 用例 |
| --- | --- | --- | --- |
| Lua 会话锁与过期写入防护 | [core/redis_lock.py:24](../src/aegis/core/redis_lock.py#L24) `RedisSessionLock.acquire/release`；[agent/state.py:80](../src/aegis/agent/state.py#L80) `RedisStateStore.save` | `SET NX EX` 获取随机令牌，Lua 比较后删除；保存 FSM 时再次检查令牌。mock 默认租期 15s，不自动续租；真实模型按工作流预算扩展租期 | [test_redis_lock.py](../tests/unit/test_redis_lock.py) `test_expired_owner_cannot_delete_new_owner`；[test_workflow.py](../tests/unit/test_workflow.py) `test_stale_lock_owner_cannot_overwrite_state_or_delete_new_lock`；[test_redis_resilience.py](../tests/integration/test_redis_resilience.py) |
| 滑动窗口三态熔断 | [core/circuit_breaker.py:123](../src/aegis/core/circuit_breaker.py#L123) `RedisCircuitBreaker.call`、`_ADMIT_SCRIPT`、`_FINISH_SCRIPT` | 默认 30s 窗口内 5 次失败触发 OPEN，15s 冷却后进入 HALF_OPEN，默认只准入 1 个探测；Redis TIME 与 generation 防止跨节点时钟偏差和迟到回调污染新状态 | [test_circuit_breaker.py](../tests/unit/test_circuit_breaker.py) `test_half_open_allows_one_probe_across_instances`、`test_late_closed_failure_cannot_pollute_new_generation`；[test_redis_resilience.py](../tests/integration/test_redis_resilience.py) |
| SHA-256 增量判定与 Alias 切换 | [rag/parser.py:340](../src/aegis/rag/parser.py#L340) `MarkdownParser.parse`；[rag/syncer.py:26](../src/aegis/rag/syncer.py#L26) `publish_snapshot`；[rag/indexer.py:73](../src/aegis/rag/indexer.py#L73) `QdrantIndexer.switch_alias` | 完全相同的快照跳过向量生成；任一变更重建完整快照。先提交父块，再切 Alias，保留旧集合/父块；没有跨 MySQL / Qdrant 分布式事务 | [test_kb_sync.py](../tests/integration/test_kb_sync.py) `test_sku_rows_persist_in_mysql_and_identical_snapshot_skips_embedding`、`test_alias_publish_failure_preserves_reads_and_retry_repairs_publication` |
| 父子切片及 SKU 表格防截断 | [rag/parser.py:267](../src/aegis/rag/parser.py#L267) `_table_chunks`；[rag/parser.py:326](../src/aegis/rag/parser.py#L326) `MarkdownParser` | H1–H3；2000 / 400 / 50 均按字符计，不是 Token。重复表头、完整行及 `sku_rows_json` 同步保留；超长单行允许突破目标长度，表格不作字符级重叠 | [test_parser.py](../tests/unit/test_parser.py) `test_large_sku_table_has_complete_rows_repeated_headers_and_structured_payload`、`test_single_oversized_table_row_is_never_cut` |
| Dense + BM25 Sparse + RRF + 600ms 降级 | [rag/sparse.py:33](../src/aegis/rag/sparse.py#L33) `encode_document/encode_query`；[rag/retriever.py:96](../src/aegis/rag/retriever.py#L96) `reciprocal_rank_fusion`；[rag/retriever.py:159](../src/aegis/rag/retriever.py#L159) `HybridRetriever.search` | 命名向量 `dense`、`bm25_sparse`；TaskGroup 并发召回；父块去重，RRF k=60；仅精排受 `wait_for(..., 0.6)` 预算约束，超时/异常回退原 RRF 并记 `rerank_fallback` | [test_retriever.py](../tests/integration/test_retriever.py) `test_rerank_timeout_returns_original_rrf_and_persists_audit`、`test_real_qdrant_hybrid_acl_timeout_and_mysql_audit` |
| 前置 ACL 与提示词越权隔离 | [api/v1/auth.py:16](../src/aegis/api/v1/auth.py#L16) `resolve_access`；[rag/retriever.py:79](../src/aegis/rag/retriever.py#L79) `build_acl_filter`；[guardrails/access_policy.py:21](../src/aegis/guardrails/access_policy.py#L21) `is_instruction_override` | Bearer 映射服务端身份，部门集合与最低职级同时限制两路召回；提示词、角色请求头都不能提升权限。当前是身份/部门/职级隔离，不应宣称存在独立 `tenant_id` 数据分片 | [test_gateway.py](../tests/unit/test_gateway.py) `test_bearer_identity_wins_over_forged_headers_and_body_is_strict`；[test_m4_gateway.py](../tests/integration/test_m4_gateway.py) `test_finance_acl_blocks_low_role_and_cross_department_without_leakage` |
| 高危词 P0 前置短路 | [guardrails/precheck.py:9](../src/aegis/guardrails/precheck.py#L9) `HighRiskPrecheck`；[agent/workflow.py:182](../src/aegis/agent/workflow.py#L182) `ControlledWorkflow._execute_node` 的 PRECHECK 分支 | 必须保留漏电、冒烟、起火、批量资损、监管投诉；先做规范化匹配，命中直接 P0，连 FSM 读取和摘要也跳过。「0ms」只能解释为零模型等待，不能宣称数据库建单零耗时 | [test_workflow.py](../tests/unit/test_workflow.py) `test_high_risk_short_circuits_all_models_retrieval_and_state`；[test_agent_workflow.py](../tests/integration/test_agent_workflow.py) `test_high_risk_zero_model_and_idempotent_ticket_real_storage` |
| 8 步 / 2 轮 / 10 轮状态机 | [agent/state.py:23](../src/aegis/agent/state.py#L23) `StepBudget`；[agent/workflow.py:182](../src/aegis/agent/workflow.py#L182) `_execute_node`；[agent/workflow.py:446](../src/aegis/agent/workflow.py#L446) `_finish` | 每次请求最多 8 个节点；排障槽位每轮最多问 2 项、最多 2 轮；每累计 10 轮压缩摘要并保留最近 4 条消息；超预算 P1，槽位耗尽 P2 | [test_workflow.py](../tests/unit/test_workflow.py) `test_eight_step_hard_limit_terminates_a_bad_node_cycle`、`test_two_slot_questions_per_turn_and_two_round_limit_hands_off`、`test_tenth_turn_compresses_and_prunes_history_then_waits_for_next_window`；[test_agent_workflow.py](../tests/integration/test_agent_workflow.py) |
| 量纲归一化 + 1 次重生成 | [guardrails/sku_verifier.py:107](../src/aegis/guardrails/sku_verifier.py#L107) `extract_quantities`；[guardrails/sku_verifier.py:246](../src/aegis/guardrails/sku_verifier.py#L246) `SKUVerifier.verify`；[agent/workflow.py:182](../src/aegis/agent/workflow.py#L182) VERIFY / REGENERATE | Decimal 单位换算；按 SKU、属性和局部语境对齐，禁止借其他型号/属性的正确数值。第二次仍错误则只返回安全文案，并记录 `SKU_HALLUCINATION_BLOCKED` / `HALLUCINATION`，建 P1 | [test_guardrails.py](../tests/unit/test_guardrails.py) `test_unit_equivalence_and_exact_ranges_pass`、`test_wrong_value_dimension_other_row_and_truncated_range_fail`；[test_workflow.py](../tests/unit/test_workflow.py) `test_repeated_sku_hallucination_never_reaches_response_or_history` |
| 24h 幂等工单与事务发件箱 | [workers/tickets.py:75](../src/aegis/workers/tickets.py#L75) `TicketService.create/publish_pending`；[db/models.py:87](../src/aegis/db/models.py#L87) `OpsWorkOrder`；[db/models.py:146](../src/aegis/db/models.py#L146) `OpsTicketOutbox` | `idem:ticket:{key}` 缓存 86400s；数据库唯一键在缓存过期后仍防重；工单与 outbox 同事务，Redis 失败可补发。事件发布标记不是 24h 缓存，恢复时须与流一起保留 | [test_ticket_worker.py](../tests/integration/test_ticket_worker.py) `test_real_concurrent_idempotency_24h_cache_and_normal_dispatch`、`test_real_redis_publish_outage_is_recovered_from_mysql_outbox` |
| Redis Streams、重试与 DLQ | [workers/consumer.py:57](../src/aegis/workers/consumer.py#L57) `TicketWorker`、`_FAILURE_SCRIPT`、`_read_reclaimed` | 消费组 `cg_ops_worker`；PEL 与 XAUTOCLAIM 恢复；累计 3 次处理失败（含首次）迁入 DLQ 并 ACK，不是「首次失败后再重试 3 次」。默认只更新 MySQL 派单状态，外部履约处理器须自行实现业务幂等 | [test_ticket_worker.py](../tests/integration/test_ticket_worker.py) `test_real_three_failures_move_full_event_to_dlq_and_ack`、`test_real_stale_pending_reclaimed_and_ack_loss_does_not_repeat_business_update` |
| OpenAPI 3.1 + FastMCP 三工具 | [main.py:28](../src/aegis/main.py#L28) `AegisFastAPI`；[api/v1/gateway.py:23](../src/aegis/api/v1/gateway.py#L23) REST 路由；[mcp/server.py:61](../src/aegis/mcp/server.py#L61) `create_mcp_server` | `search_enterprise_kb`、`verify_sku_specs`、`create_ops_ticket` 共用 `GatewayServices`，挂载 `/mcp/sse`；握手与消息投递都鉴权，跨身份 SSE 会话投递被拒绝 | [test_gateway.py](../tests/unit/test_gateway.py) `test_openapi_has_31_typed_business_contracts_and_unique_operations`；[test_mcp_gateway.py](../tests/unit/test_mcp_gateway.py) `test_mcp_rejects_unauthenticated_and_cross_identity_messages`；[test_m4_gateway.py](../tests/integration/test_m4_gateway.py) |
| Semaphore 并发评测与双账本 | [eval/runner.py:93](../src/aegis/eval/runner.py#L93) `EvaluationRunner`；[eval/runner.py:377](../src/aegis/eval/runner.py#L377) `run`；[eval/judge.py:96](../src/aegis/eval/judge.py#L96) `AppTokenLedger/JudgeTokenLedger` | Semaphore 限流，gather 保持输入顺序；业务时延与裁判耗时分开，应用/裁判用量独立；缺样本、执行异常、红线不达标均阻断门禁 | [test_eval_runner.py](../tests/unit/test_eval_runner.py) `test_semaphore_bounds_peak_and_preserves_input_order`、`test_app_judge_ledgers_are_separate_and_p0_does_not_dilute_retrieval_metrics`；[test_eval_pipeline.py](../tests/integration/test_eval_pipeline.py) |
| 百分比容差裁判 | [eval/judge.py:234](../src/aegis/eval/judge.py#L234) `NumericJudge.equal/compare`；[eval/judge.py:473](../src/aegis/eval/judge.py#L473) `AnswerJudge` | 默认百分比绝对容差 0.15 个百分点，其他数值相对容差 0.001；`0.0523` 与 `5.2%` 差 0.03 个百分点可通过。该容差属于评测裁判，不能移植到 SKU 在线严格核验 | [test_eval_judge.py](../tests/unit/test_eval_judge.py) `test_numeric_percentage_and_units`、`test_binding_failure_prevents_real_judge_from_overriding_wrong_assignment` |
| 三级瀑布归因 | [eval/attribution.py:10](../src/aegis/eval/attribution.py#L10) `attribute_failure`；[eval/runner.py:182](../src/aegis/eval/runner.py#L182) `_evaluate` | 只用同次业务轨迹：粗召回缺失→MISSED_RECALL；最终上下文缺失→LOW_RANKING；上下文命中但回答失败→HALLUCINATION。仅对黄金常规样本归因，P0 / ACL 独立验收 | [test_eval_attribution.py](../tests/unit/test_eval_attribution.py) `test_waterfall_all_branches`；[test_eval_runner.py](../tests/unit/test_eval_runner.py) `test_redteam_failure_is_not_misclassified_as_missing_secret_recall` |

## 三、Top 8 高频追问与源码级答法

### 1. 「Redis 锁 15 秒到期，旧请求还在运行，会不会覆盖新请求？」

**口语答法：**「我把互斥和过期写入分开处理。加锁是 `SET NX EX`，值是每次随机生成的所有权令牌。释放时用 Lua 比较令牌，所以旧持有者不会删掉新锁。更关键的是，写 FSM 的 Lua 也检查同一令牌，租期失效后就拒绝落状态。mock 默认 10 秒执行预算，加 4 秒收尾预算，锁设为 15 秒；真实模型的预算会相应扩展。这是固定租期设计，不是看门狗续租。它保护的是会话状态，工单副作用还要依赖数据库唯一键保证幂等。」

**现场定位：** `RedisSessionLock.release`、`RedisStateStore.save`、`ControlledWorkflow.__init__`、`agent/runtime.py:create_workflow`。可运行过期锁所有者测试，证明释放与写入两个边界都受保护。

### 2. 「为什么选 Dense + BM25 + RRF？600ms 能保证整条请求返回吗？」

**口语答法：**「仓库里既有语义性的 SOP，也有 SKU 编码、型号和规则编号。Dense 与稀疏词项匹配互补，但两路分数尺度不同，所以我按父块去重后累加 `1/(60+rank)`。Rerank 用来改善最后的上下文顺序，不能成为可用性单点。代码仅对精排包 0.6 秒 `asyncio.wait_for`，发生超时、接口异常或返回分数不合法时，保留原 RRF 结果，并写审计降级字段。600ms 是精排协程预算，端到端还包括 embedding、检索和数据库提交；异步取消也依赖供应商协程及时响应，不能把它说成整个 HTTP 请求的绝对时限。」

**现场定位：** `HybridRetriever.search`、`reciprocal_rank_fusion`、`validate_scores`。故障演示中对照 850ms 注入、精排取消时刻、实际端到端耗时与审计 ID。

### 3. 「知识库更新怎样做到不停机？MySQL 提交了、Alias 切换失败怎么办？」

**口语答法：**「我发布的是完整快照。文档 SHA-256、切片 ID、embedding 指纹、维度和稀疏版本共同决定物理集合名，相同快照且数量吻合才跳过构建。新集合准备完成后先提交 MySQL 父块，再原子替换 Qdrant Alias，旧集合和旧父块保留。在途检索先把 Alias 解析成一个物理集合，Dense 和 Sparse 都查询它，因此不会混两个版本。跨库没有分布式事务；切换失败时旧 Alias 继续可读，重跑修复发布状态。旧版本回收需要单独管理，当前没有承诺自动清理。」

**现场定位：** `publish_snapshot`、`QdrantIndexer.switch_alias`。特别说明「增量」指相同快照跳过，不是变更单个文档后仅重算该文档。

### 4. 「提示词说自己是财务总监，你如何保证财务数据不进模型？」

**口语答法：**「权限来自服务端 Bearer 映射的 `AccessContext`，不会从用户话术或角色请求头采信。两路 Qdrant 查询都带部门集合和 `min_role_level <= role_level`，缺权限直接报错，不会把过滤条件省掉。可识别的越权指令还有前置拒绝策略，但真正的数据边界是认证和检索过滤。独立 SKU 核验接口也会先检查父块权限，不能绕过检索拿任意父块来核验。评测同时检查粗召回和最终上下文中的禁访文档，再检查上下文与回答中的敏感词；只看最后答案是否拒答是不够的。」

**现场定位：** `resolve_access`、`build_acl_filter`、`GatewayServices.verify_sku`。当前没有独立 `tenant_id` 字段，应称为身份、部门与职级隔离，不把仓库未实现的租户分库写入简历。

### 5. 「850g 改成 0.85kg 怎么放行？模型把 2100W 改成 2200W 怎么拦？」

**口语答法：**「我先提取数量，用 Decimal 统一到基准量纲，再把主张绑定到 SKU 和属性。等价单位会归一到同一个值；同一个数字属于另一个型号、净重和毛重混用、温度上限和范围混用都不能借值通过。真值只从授权父块对应的 MySQL 结构化行读取。第一次错误会返回具体真值纠正提示，工作流只允许重生成一次；第二次仍失败，就用固定安全文案替换候选答案，写幻觉审计并建 P1 工单，错误参数不会进入响应或会话历史。评测中的百分比容差是另一套规则，在线 SKU 护栏没有这个宽松容差。」

**现场定位：** `SKUVerifier.verify/_equal`、`load_sku_rows`、工作流的 VERIFY / REGENERATE，以及 `test_repeated_sku_hallucination_never_reaches_response_or_history`。

### 6. 「Redis 宕机、用户连点、消费者处理完却没 ACK，会不会重复派单？」

**口语答法：**「先区分建单和消费。建单的事实源是 MySQL，幂等键有唯一约束，Redis 24 小时缓存只是快速返回。并发冲突后用新事务读回同一工单，避免复用旧事务快照。工单和 outbox 一起提交；Redis 不可用时保留待发布事件，恢复后补发。发布时 Lua 原子执行事件去重标记和 XADD。消费者用 PEL 和 XAUTOCLAIM 恢复未确认消息，默认派单在数据库行锁下只把 PENDING 改成 DISPATCHED，重投不会重复改状态。累计三次处理失败后 Lua 原子写 DLQ 并 ACK。这个链路不能宣称所有外部系统 exactly-once：如果接真实 WMS，外部处理器必须使用业务幂等键。」

**现场定位：** `TicketService.create/publish_pending`、`TicketWorker.dispatch/_process_message`。补充：Redis 发布标记和 Stream 需要一起持久化、一起恢复，不能只删除去重标记。

### 7. 「为什么限制 8 步、2 轮、10 轮？强行截断会不会丢业务？」

**口语答法：**「这是确定性受控工作流，不让模型自由决定是否无限调用工具。每个节点进入前消耗一步预算，正常路径、一次纠错路径都有明确终点。排障信息不足时每轮最多补问两项，最多两轮；仍不完整就带已有槽位和完整度建 P2 工单。每十轮压缩历史，保留近期消息，摘要失败则用有界原文摘录，不让上下文无限增长。循环超限、执行超时或依赖故障会转 P1；P0 是更早的独立分支，既不读 FSM，也不调用检索、生成或摘要。P0 建单失败会显式报错，不能伪装成已创建低优先级工单。」

**现场定位：** `StepBudget.enter`、`_execute_node`、`_finish`、`_fallback`。说明「零模型调用」不等于端到端零耗时，实际建单和审计仍有 I/O。

### 8. 「为什么你的通过率是 100%？怎么证明不是测评自嗨？」

**口语答法：**「我会先交代样本与运行模式：这是 40 条黄金问答和 13 条红队、真实本地存储、确定性 mock 应用与 mock 裁判的回归结果，不是线上真实模型效果。检索指标只算 34 条普通黄金样本，P0 不参与分母；文档召回全中也不能掩盖父块 Recall 只有 78.63%。父块分母来自完整授权库，不从已召回集合反推。`0.0523` 和 `5.2%` 会换成同一比例再比较，差值是 0.03 个百分点，低于默认 0.15 个百分点容差；商品归属和属性绑定还要独立检查。失败归因用同次工作流的粗召回、最终上下文和答案，避免二次检索覆盖现场。空数据集、执行异常、P0 或 ACL 红线失败都会阻断发布。接真实千问后应重新跑门禁，保留供应商 usage 和对应报告，而不能沿用 mock 数字。」

**现场定位：** `EvaluationRunner._expected_parents/_evaluate/run`、`NumericJudge`、`attribute_failure`。应用和裁判两份账本分别说明模型成本，不能把 mock 字符估算宣称为真实付费 Token。

## 四、面试前最后一次证据复核

1. 使用 [README](../README.md) 的启动命令准备真实容器，完成迁移和知识库初始化。
2. 执行 `uv run python scripts/demo_interview_chaos.py`，解释四幕的输入、故障注入、实时防线、审计/工单 ID 和实测耗时。
3. 执行 `uv run ruff check src tests scripts`、`AEGIS_INTEGRATION=1 uv run pytest -v` 与 `uv run python scripts/run_eval.py --enforce-gate`；演示成功不能替代全量回归与评测门禁。
4. 核对新的报告 `run_id`、运行模式、样本数与指标，再更新简历量化表述。不得沿用旧时延数字，或把未验证的吞吐、节省人力、投诉下降、GMV 增长填入成果。
5. 面试现场优先打开上述类/函数和对应测试，用代码解释边界。尚未实现的身份提供方、独立租户分片、外部 WMS 派单、自动旧快照回收，应明确作为后续扩展方向。
