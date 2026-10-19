from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.digest import router as digest_router
from app.api.health import router as health_router
from app.api.interests import router as interests_router
from app.api.papers import router as papers_router
from app.api.search import router as search_router
from app.egress_guard import install_from_env

# Proof runs only: does nothing unless EGRESS_GUARD_POLICY is set (see app/egress_guard.py).
# The hooks patch the HTTP clients' classes, so they cover clients created before or after.
install_from_env()

app = FastAPI(title="AI Research Copilot")
app.include_router(health_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(papers_router)
app.include_router(interests_router)
app.include_router(digest_router)
