"""FastAPI application entry point."""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db, close_db, async_session_maker
from .routers import personas_router, events_router, dialogues_router, tasks_router
from .services.persona import import_base_personas

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()

    # Import base personas from JSON if needed
    personas_file = Path(__file__).parent.parent.parent / "user_personas.json"
    if personas_file.exists():
        async with async_session_maker() as db:
            with open(personas_file, "r", encoding="utf-8") as f:
                personas_data = json.load(f)
            count = await import_base_personas(db, personas_data)
            await db.commit()
            if count > 0:
                print(f"Imported {count} base personas")

    yield

    # Shutdown
    await close_db()


app = FastAPI(
    title="Medical Memory Evaluation Dataset Generator",
    description="API for generating medical dialogue datasets for memory evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(personas_router)
app.include_router(events_router)
app.include_router(dialogues_router)
app.include_router(tasks_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Medical Memory Evaluation Dataset Generator",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
