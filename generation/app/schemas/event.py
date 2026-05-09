"""Event Pydantic schemas."""
from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, Field


class EventNodeResponse(BaseModel):
    """Event node response schema."""

    id: int
    event_graph_id: int
    event: str
    type: str
    event_date: str  # YYYY-MM-DD
    triggered_by: list[int] = []

    class Config:
        from_attributes = True


class EventNodeCreate(BaseModel):
    """Schema for creating an event node."""

    event: str = Field(..., description="Event description including subject info")
    type: str = Field(
        ...,
        pattern="^(health|life|work|allergy|medication_history|disease_history|medication_preference|diet_preference|lifestyle_economic)$",
        description="Event type (regular: health/life/work, trap: allergy/medication_history/disease_history/medication_preference/diet_preference/lifestyle_economic)"
    )
    event_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Event date (YYYY-MM-DD)")
    triggered_by: list[int] = Field(default_factory=list, description="Prerequisite event IDs that triggered this event")


class EventNodeUpdate(BaseModel):
    """Schema for updating an event node."""

    event: Optional[str] = None
    type: Optional[str] = Field(
        None,
        pattern="^(health|life|work|allergy|medication_history|disease_history|medication_preference|diet_preference|lifestyle_economic)$"
    )
    event_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    triggered_by: Optional[list[int]] = None


class EventGraphResponse(BaseModel):
    """Event graph response schema."""

    id: int
    expanded_persona_id: int
    start_date: str  # YYYY-MM-DD
    time_span_days: int
    event_nodes: list[EventNodeResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True


class EventGenerateRequest(BaseModel):
    """Request to generate events for a persona."""

    start_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Timeline start date (YYYY-MM-DD)",
        examples=["2024-01-01"]
    )
    time_span_days: int = Field(90, ge=7, le=365, description="Time span in days")
    min_events: int = Field(5, ge=1, le=50, description="Minimum number of events")
    max_events: int = Field(15, ge=1, le=50, description="Maximum number of events")
