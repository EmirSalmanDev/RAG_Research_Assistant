"""GET /health: actively probes Postgres, Qdrant and Redis."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

import redis.asyncio as aioredis
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings

router = APIRouter()

TIMEOUT_S = 2.0  # a hung store must not hang the healthcheck

Status = Literal["up", "down"]


class HealthResponse(BaseModel):
    postgres: Status
    qdrant: Status
    redis: Status


async def check_postgres() -> None:
    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    finally:
        await engine.dispose()


async def check_qdrant() -> None:
    client = AsyncQdrantClient(url=get_settings().qdrant_url, timeout=int(TIMEOUT_S))
    try:
        await client.get_collections()
    finally:
        await client.close()


async def check_redis() -> None:
    client = aioredis.from_url(get_settings().redis_url)
    try:
        await client.ping()
    finally:
        await client.aclose()


async def _probe(check: Callable[[], Awaitable[None]]) -> Status:
    try:
        await asyncio.wait_for(check(), timeout=TIMEOUT_S)
    except Exception:
        return "down"
    return "up"


@router.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    # Looked up at call time (module globals) so tests can monkeypatch the checks.
    pg, qd, rd = await asyncio.gather(
        _probe(check_postgres), _probe(check_qdrant), _probe(check_redis)
    )
    result = HealthResponse(postgres=pg, qdrant=qd, redis=rd)
    if "down" in result.model_dump().values():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result
