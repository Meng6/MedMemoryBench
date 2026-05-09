"""Task Pydantic schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TaskResponse(BaseModel):
    """Async task response schema."""

    id: int
    task_type: str
    status: str
    progress: float
    total_items: int
    completed_items: int
    params: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TaskStatusResponse(BaseModel):
    """Simplified task status response."""

    id: int
    status: str
    progress: float
    completed_items: int
    total_items: int
    error: Optional[str] = None
