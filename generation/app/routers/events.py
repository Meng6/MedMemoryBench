"""Event API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.event import (
    EventNodeResponse,
    EventNodeCreate,
    EventNodeUpdate,
    EventGraphResponse,
    EventGenerateRequest,
)
from ..services.event import EventService

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("/{persona_id}", response_model=EventGraphResponse)
async def get_event_graph(persona_id: int, db: AsyncSession = Depends(get_db)):
    """Get persona event graph."""
    service = EventService(db)
    graph = await service.get_event_graph(persona_id)
    if not graph:
        raise HTTPException(status_code=404, detail="Event graph not found")
    return graph


@router.post("/generate/{persona_id}", response_model=EventGraphResponse)
async def generate_events(
    persona_id: int,
    request: EventGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate event graph for persona."""
    service = EventService(db)
    try:
        graph = await service.generate_events(persona_id, request)
        return graph
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@router.post("/{persona_id}/nodes", response_model=EventNodeResponse)
async def create_event_node(
    persona_id: int,
    data: EventNodeCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create event node."""
    service = EventService(db)
    graph = await service.get_event_graph(persona_id)
    if not graph:
        raise HTTPException(status_code=404, detail="Event graph not found")

    node = await service.create_event_node(graph.id, data)
    return node


@router.put("/nodes/{node_id}", response_model=EventNodeResponse)
async def update_event_node(
    node_id: int,
    data: EventNodeUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update event node."""
    service = EventService(db)
    node = await service.update_event_node(node_id, data)
    if not node:
        raise HTTPException(status_code=404, detail="Event node not found")
    return node


@router.delete("/nodes/{node_id}")
async def delete_event_node(node_id: int, db: AsyncSession = Depends(get_db)):
    """Delete event node."""
    service = EventService(db)
    success = await service.delete_event_node(node_id)
    if not success:
        raise HTTPException(status_code=404, detail="Event node not found")
    return {"status": "ok", "message": "Event node deleted"}
