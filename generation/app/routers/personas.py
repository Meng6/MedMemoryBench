"""Persona API routes."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.persona import (
    BasePersonaResponse,
    ExpandedPersonaResponse,
    ExpandedPersonaCreate,
    ExpandedPersonaUpdate,
    PersonaExpandRequest,
    PersonaBatchExpandRequest,
)
from ..schemas.task import TaskResponse
from ..services.persona import PersonaService
from ..models import AsyncTask

router = APIRouter(prefix="/api/personas", tags=["personas"])


@router.get("/base", response_model=list[BasePersonaResponse])
async def get_base_personas(db: AsyncSession = Depends(get_db)):
    """Get all base personas."""
    service = PersonaService(db)
    personas = await service.get_all_base_personas()
    return personas


@router.get("/base/{persona_id}", response_model=BasePersonaResponse)
async def get_base_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single base persona."""
    service = PersonaService(db)
    persona = await service.get_base_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.get("/expanded", response_model=list[ExpandedPersonaResponse])
async def get_expanded_personas(db: AsyncSession = Depends(get_db)):
    """Get all expanded personas."""
    service = PersonaService(db)
    personas = await service.get_all_expanded_personas()
    return personas


@router.get("/expanded/{persona_id}", response_model=ExpandedPersonaResponse)
async def get_expanded_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single expanded persona."""
    service = PersonaService(db)
    persona = await service.get_expanded_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Expanded persona not found")
    return persona


@router.post("/expanded", response_model=ExpandedPersonaResponse)
async def create_expanded_persona(
    data: ExpandedPersonaCreate, db: AsyncSession = Depends(get_db)
):
    """Manually create an expanded persona."""
    service = PersonaService(db)

    # Verify base persona exists
    base = await service.get_base_persona(data.base_persona_id)
    if not base:
        raise HTTPException(status_code=404, detail="Base persona not found")

    persona = await service.create_expanded_persona(data)
    return persona


@router.put("/expanded/{persona_id}", response_model=ExpandedPersonaResponse)
async def update_expanded_persona(
    persona_id: int,
    data: ExpandedPersonaUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an expanded persona."""
    service = PersonaService(db)
    persona = await service.update_expanded_persona(persona_id, data)
    if not persona:
        raise HTTPException(status_code=404, detail="Expanded persona not found")
    return persona


@router.delete("/expanded/{persona_id}")
async def delete_expanded_persona(persona_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an expanded persona."""
    service = PersonaService(db)
    success = await service.delete_expanded_persona(persona_id)
    if not success:
        raise HTTPException(status_code=404, detail="Expanded persona not found")
    return {"status": "ok", "message": "Persona deleted"}


@router.post("/expand", response_model=ExpandedPersonaResponse)
async def expand_persona(
    request: PersonaExpandRequest, db: AsyncSession = Depends(get_db)
):
    """Expand a single persona using LLM."""
    service = PersonaService(db)
    try:
        persona = await service.expand_persona(request)
        return persona
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


async def _batch_expand_task(
    task_id: int,
    request: PersonaBatchExpandRequest,
    db_url: str,
):
    """Background task for batch persona expansion."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    engine = create_async_engine(db_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as db:
        # Update task status
        from sqlalchemy import select, update
        await db.execute(
            update(AsyncTask)
            .where(AsyncTask.id == task_id)
            .values(status="running")
        )
        await db.commit()

        service = PersonaService(db)
        total = len(request.base_persona_ids) * request.count_per_persona
        completed = 0
        results = []
        errors = []

        for base_id in request.base_persona_ids:
            for i in range(request.count_per_persona):
                try:
                    expand_req = PersonaExpandRequest(base_persona_id=base_id)
                    persona = await service.expand_persona(expand_req)
                    await db.commit()
                    results.append(persona.id)
                except Exception as e:
                    errors.append({"base_persona_id": base_id, "error": str(e)})

                completed += 1
                progress = completed / total

                await db.execute(
                    update(AsyncTask)
                    .where(AsyncTask.id == task_id)
                    .values(
                        progress=progress,
                        completed_items=completed,
                    )
                )
                await db.commit()

        # Final update
        await db.execute(
            update(AsyncTask)
            .where(AsyncTask.id == task_id)
            .values(
                status="completed" if not errors else "completed",
                progress=1.0,
                result={"expanded_persona_ids": results, "errors": errors},
            )
        )
        await db.commit()

    await engine.dispose()


@router.post("/expand/batch", response_model=TaskResponse)
async def batch_expand_personas(
    request: PersonaBatchExpandRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Batch expand personas (async task)."""
    from ..config import get_settings

    settings = get_settings()

    # Verify all base personas exist
    service = PersonaService(db)
    for base_id in request.base_persona_ids:
        base = await service.get_base_persona(base_id)
        if not base:
            raise HTTPException(
                status_code=404, detail=f"Base persona {base_id} not found"
            )

    # Create async task
    task = AsyncTask(
        task_type="expand_batch",
        status="pending",
        total_items=len(request.base_persona_ids) * request.count_per_persona,
        params=request.model_dump(),
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    await db.commit()

    # Start background task
    background_tasks.add_task(
        _batch_expand_task,
        task.id,
        request,
        settings.database_url,
    )

    return task
