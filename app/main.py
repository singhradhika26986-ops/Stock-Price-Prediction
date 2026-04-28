from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging
from app.middleware import MonitoringMiddleware, RateLimitMiddleware
from app.services.scheduler import retraining_scheduler

configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    retraining_scheduler.start()
    try:
        yield
    finally:
        retraining_scheduler.stop()


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(MonitoringMiddleware)
app.add_middleware(RateLimitMiddleware)
app.include_router(router)
