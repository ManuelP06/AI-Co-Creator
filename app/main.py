from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core import setup_logging
from app.core.cache import init_redis
from app.core.middleware import (ExceptionHandlerMiddleware, LoggingMiddleware,
                                 SecurityHeadersMiddleware)
from app.routers import (analysis_router, editor_router, pipeline_router,
                         renderer_router, shots_router, transcription_router,
                         upload_router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    await init_redis()
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(LoggingMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload_router.router, prefix="/api/v1")
app.include_router(shots_router.router, prefix="/api/v1")
app.include_router(transcription_router.router, prefix="/api/v1")
app.include_router(analysis_router.router, prefix="/api/v1")
app.include_router(editor_router.router, prefix="/api/v1")
app.include_router(renderer_router.router, prefix="/api/v1")
app.include_router(pipeline_router.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Co-Creator API",
        "version": settings.api_version,
        "docs": "/docs" if settings.debug else "Documentation disabled in production",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.api_version}
