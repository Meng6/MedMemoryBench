"""Persona Pydantic schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .persona_enriched import (
    EnrichedFieldsSchema,
    LifestyleSchema,
    HealthDetailsSchema,
)


class BasePersonaResponse(BaseModel):
    """Base persona response schema."""

    id: int
    type_name: str
    gender: str
    core_feature: Optional[str] = None
    health_goals: Optional[list[str]] = None
    category: Optional[str] = None

    class Config:
        from_attributes = True


class ExpandedPersonaResponse(BaseModel):
    """Expanded persona response schema."""

    id: int
    base_persona_id: int
    enriched_data: Optional[EnrichedFieldsSchema] = None
    created_at: datetime

    # Include base persona info
    base_persona: Optional[BasePersonaResponse] = None

    class Config:
        from_attributes = True


class ExpandedPersonaCreate(BaseModel):
    """Schema for creating an expanded persona manually."""

    base_persona_id: int
    enriched_data: Optional[EnrichedFieldsSchema] = None


class ExpandedPersonaUpdate(BaseModel):
    """Schema for updating an expanded persona."""

    enriched_data: Optional[EnrichedFieldsSchema] = None


class PersonaExpandRequest(BaseModel):
    """Request to expand a single persona using LLM."""

    base_persona_id: int


class PersonaBatchExpandRequest(BaseModel):
    """Request to expand multiple personas in batch."""

    base_persona_ids: list[int] = Field(..., description="List of base persona IDs to expand")
    count_per_persona: int = Field(1, ge=1, le=10, description="Number of expansions per persona")
