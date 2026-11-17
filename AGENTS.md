# AGENTS.md — Engineering Constitution for `aegis-rag-opsflow`

## 1. Project Identity & Mission
`aegis-rag-opsflow` is a production-grade Enterprise Multi-Tenant Hybrid RAG Platform & Deterministic Fulfillment Work-Order Agent System tailored for e-commerce operations, warehouse SOPs, and SKU specification verification.

Every module written in this repository MUST survive senior architect code reviews and live production failure modes (network jitter, LLM hallucinations, concurrent double-clicks, unauthorized cross-department queries, and zero-downtime re-indexing).

---

## 2. Technology Stack & Runtime Rules
- **Language & Package Manager**: Python `>=3.12`, managed strictly via `uv` (`pyproject.toml` + `uv.lock`). NEVER use raw `pip install` without updating `pyproject.toml`.
- **Web & Protocol Layer**: `FastAPI` (OpenAPI 3.1 compliant) + `FastMCP` (`/mcp/sse` standard tool exposure).
- **Storage & Infrastructure (Docker Compose)**:
  - `MySQL 8.0`: Accessed ONLY via `SQLAlchemy 2.0` async engine (`asyncmy` driver) + `Alembic` migrations.
  - `Redis 7`: Accessed via `redis.asyncio` for distributed session locks, finite-state machine (FSM) context, idempotency keys, circuit breaker counters, and `Redis Streams` (`cg_ops_worker` + Dead-Letter Queue).
  - `Qdrant`: Accessed via `AsyncQdrantClient` using Named Vectors (`dense` + `bm25_sparse`), Payload ACL indexes, and Collection Aliases (`kb_children_prod`) for atomic zero-downtime switching.
- **Observability**: `structlog` (JSON structured logs with `trace_id` via `contextvars`) + `prometheus-client` (`/metrics`).
- **Quality & Testing**: `ruff` (lint + format), `pytest`, `pytest-asyncio`.

---

## 3. Standard Repository Directory Layout
All code MUST follow this exact layered architecture (do not invent ad-hoc top-level folders):

```text
aegis-rag-opsflow/
├── AGENTS.md
├── CLAUDE.md                  # Symlink to AGENTS.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── alembic.ini
├── alembic/                   # Database migrations
├── scripts/                   # Seed data, alias switch CLI, load test scripts
├── data/
│   ├── seed_docs/             # Realistic e-commerce SKU tables, SOPs, after-sales rules
│   └── golden_eval/           # Golden QA & ACL red-line benchmark datasets
├── src/
│   └── aegis/
│       ├── main.py            # FastAPI lifespan, routers, MCP mount, middlewares
│       ├── config.py          # pydantic-settings strict configuration
│       ├── core/              # trace_context, structlog, exceptions, circuit_breaker, redis_lock
│       ├── db/                # SQLAlchemy 2.0 async session, ORM models, repositories
│       ├── schemas/           # Pydantic v2 request/response contracts (OpenAPI 3.1)
│       ├── llm/               # Unified async LLM/Embedding/Rerank provider (DashScope/OpenAI + Deterministic Mock mode)
│       ├── rag/               # Markdown H1-H3 parser, table protector, SHA-256 syncer, Qdrant hybrid + RRF + Rerank + Alias
│       ├── guardrails/        # High-risk keyword short-circuit, SKU row-level unit verifier
│       ├── agent/             # Redis-backed controlled loop (<=8 steps), slot filling (<=2 rounds), 10-turn compressor
│       ├── workers/           # Redis Streams async ticket consumer, retry & DLQ handler
│       ├── eval/              # Semaphore runner, Hit@K/MRR metrics, numeric tolerance judge, 3-tier attribution
│       ├── mcp/               # FastMCP server tools (search_enterprise_kb, verify_sku_specs, create_ops_ticket)
│       └── api/v1/            # REST endpoints (/kb, /agent, /eval, /health)
└── tests/
    ├── unit/
    └── integration/
