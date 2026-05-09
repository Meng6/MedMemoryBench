"""Database configuration and session management."""
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from .config import get_settings

settings = get_settings()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    connect_args={
        "timeout": 30,
        "check_same_thread": False,
    },
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Alias for background tasks
AsyncSessionLocal = async_session_maker


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


async def get_db() -> AsyncSession:
    """Dependency to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables and enable WAL mode for SQLite."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if "sqlite" in settings.database_url:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))


async def close_db():
    """Close database connection."""
    await engine.dispose()
