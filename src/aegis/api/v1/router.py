"""首期 REST 路由聚合入口。"""

from fastapi import APIRouter

from aegis.api.v1.agent import router as agent_router
from aegis.api.v1.gateway import router as gateway_router
from aegis.api.v1.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(agent_router)
router.include_router(gateway_router)
