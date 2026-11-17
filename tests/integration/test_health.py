"""使用真实基础设施验证应用就绪探针和指标导出。"""

import os

import httpx
import pytest

from aegis.main import create_app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("AEGIS_INTEGRATION") != "1", reason="需设置 AEGIS_INTEGRATION=1 并启动基础设施"
    ),
]


async def test_real_readiness_and_metrics():
    application = create_app()
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            ready = await client.get("/health/ready")
            assert ready.status_code == 200, ready.text
            assert all(item["status"] == "ok" for item in ready.json()["components"].values())
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "text/plain" in metrics.headers["content-type"]
            assert 'route="/health/ready"' in metrics.text
