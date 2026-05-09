"""Task API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.task import TaskResponse, TaskStatusResponse
from ..models import AsyncTask

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("", response_model=list[TaskResponse])
async def get_tasks(db: AsyncSession = Depends(get_db)):
    """Get all async tasks."""
    result = await db.execute(
        select(AsyncTask).order_by(AsyncTask.created_at.desc())
    )
    tasks = result.scalars().all()
    return list(tasks)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Get task details."""
    result = await db.execute(
        select(AsyncTask).where(AsyncTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: int, db: AsyncSession = Depends(get_db)):
    """Get task status (lightweight)."""
    result = await db.execute(
        select(AsyncTask).where(AsyncTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        id=task.id,
        status=task.status,
        progress=task.progress,
        completed_items=task.completed_items,
        total_items=task.total_items,
        error=task.error,
    )


@router.delete("/{task_id}")
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a task record."""
    result = await db.execute(
        select(AsyncTask).where(AsyncTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Only allow deletion of completed/failed tasks
    if task.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail="Can only delete completed or failed tasks",
        )

    await db.delete(task)
    return {"status": "ok", "message": "Task deleted"}
