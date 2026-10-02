from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.search import router as search_router

app = FastAPI(title="AI Research Copilot")
app.include_router(health_router)
app.include_router(search_router)
