"""将种子文档清单作为完整知识库快照导入并发布，权限按文档逐份声明。"""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from aegis.config import Settings
from aegis.db.models import DocumentType
from aegis.db.qdrant_client import create_qdrant_client
from aegis.db.session import create_mysql_engine
from aegis.llm.providers import create_embedding_provider
from aegis.rag.parser import DocumentMetadata, MarkdownParser
from aegis.rag.syncer import publish_snapshot

# 种子清单：(文件名, doc_code, doc_type, dept_scope, min_role_level)，评测集校验复用本清单。
# 权限逐份声明：财务制度仅 FINANCE 部门角色等级 3 及以上可召回，
# 一线客服与仓储岗（warehouse / customer_service，等级 1）被 ACL 双重拒绝。
SEED_MANIFEST: tuple[tuple[str, str, DocumentType, str, int], ...] = (
    ("sku_spec_sample.md", "SEED-SKU-SPEC", DocumentType.SKU_SPEC, "warehouse", 1),
    (
        "warehouse_sop_sample.md",
        "SEED-WAREHOUSE-SOP",
        DocumentType.WAREHOUSE_SOP,
        "warehouse",
        1,
    ),
    ("ec_sku_specs.md", "EC-SKU-SPECS", DocumentType.SKU_SPEC, "warehouse", 1),
    ("ec_warehouse_sop.md", "EC-WAREHOUSE-SOP", DocumentType.WAREHOUSE_SOP, "warehouse", 1),
    (
        "ec_after_sales_rules.md",
        "EC-AFTER-SALES-RULES",
        DocumentType.AFTER_SALES,
        "customer_service",
        1,
    ),
    ("ec_finance_internal.md", "EC-FINANCE-INTERNAL", DocumentType.FINANCE_RULE, "FINANCE", 3),
)


async def initialize(seed_dir: Path) -> None:
    settings = Settings()
    parser = MarkdownParser()
    declared = {filename for filename, *_ in SEED_MANIFEST}
    discovered = await asyncio.to_thread(lambda: {path.name for path in seed_dir.glob("*.md")})
    if declared != discovered:
        # 未声明 ACL 的新文档必须显式登记，不能静默漏导或继承公共权限。
        raise ValueError(
            f"种子清单不完整：缺失文件={sorted(declared - discovered)}，"
            f"未声明权限文件={sorted(discovered - declared)}"
        )
    documents = [
        parser.parse(
            (seed_dir / filename).read_text(encoding="utf-8"),
            DocumentMetadata(
                doc_code=code,
                doc_type=doc_type,
                dept_scope=dept_scope,
                min_role_level=min_role_level,
            ),
        )
        for filename, code, doc_type, dept_scope, min_role_level in SEED_MANIFEST
    ]
    engine = create_mysql_engine(settings)
    client = create_qdrant_client(settings)
    provider = create_embedding_provider(settings)
    try:
        result = await publish_snapshot(
            documents,
            engine=engine,
            client=client,
            embedding_provider=provider,
            embedding_fingerprint=json.dumps(
                [
                    settings.embedding_provider_mode.value,
                    settings.embedding_model,
                    settings.embedding_base_url,
                    "mock-v1",
                ]
            ),
            batch_size=settings.embedding_batch_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False))
    finally:
        await provider.aclose()
        await client.close()
        await engine.dispose()


def main() -> None:
    cli = argparse.ArgumentParser(description="构建种子文档清单的完整快照并原子发布线上别名")
    cli.add_argument(
        "--seed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "seed_docs",
        help="包含种子清单全部 Markdown 文件的目录（清单见脚本内 SEED_MANIFEST）",
    )
    args = cli.parse_args()
    asyncio.run(initialize(args.seed_dir))


if __name__ == "__main__":
    main()
