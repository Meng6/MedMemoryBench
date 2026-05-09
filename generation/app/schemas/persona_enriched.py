"""Pydantic schemas for enriched user personas."""

from typing import Optional
from pydantic import BaseModel, Field


class LifestyleSchema(BaseModel):
    """Lifestyle information schema."""

    sleep_pattern: str = Field(..., description="Sleep pattern, e.g. '23:00-7:00'")
    diet_habits: str = Field(..., description="Dietary habits description")
    exercise_frequency: str = Field(..., description="Exercise frequency, e.g. 'runs 3 times/week'")
    stress_level: str = Field(..., description="Subjective stress level, e.g. 'medium-high'")


class DiseaseProgressionSchema(BaseModel):
    """Disease progression phases (aligned with deep-research report)."""

    phase_1: str = Field(
        default="",
        description="Phase 1: Misdiagnosis & compensation (~0-20 sessions)"
    )
    phase_2: str = Field(
        default="",
        description="Phase 2: Turning point (~20-40 sessions)"
    )
    phase_3: str = Field(
        default="",
        description="Phase 3: Complications (~40-60 sessions)"
    )
    phase_4: str = Field(
        default="",
        description="Phase 4: Lifestyle & psychological challenges (~60-80 sessions)"
    )
    phase_5: str = Field(
        default="",
        description="Phase 5: Follow-up & improvement (~80+ sessions)"
    )


class HealthDetailsSchema(BaseModel):
    """Health details schema."""

    medical_history: list[str] = Field(default_factory=list, description="Medical history list")
    disease_progression: Optional[DiseaseProgressionSchema] = Field(
        default=None,
        description="Disease progression phases (aligned with deep-research report)"
    )


class EnrichedFieldsSchema(BaseModel):
    """All enriched fields for a persona."""

    age_range: str = Field(..., description="Age range, e.g. '35-40'")
    occupation_detail: str = Field(..., description="Occupation detail")

    lifestyle: LifestyleSchema = Field(..., description="Lifestyle information")
    health_details: HealthDetailsSchema = Field(..., description="Health details")

    background_story: str = Field(..., description="Background story")


class BasePersonaInput(BaseModel):
    """Input schema for base persona (from user_personas.json)."""

    id: int
    type_name: str
    gender: str
    core_feature: str
    health_goals: list[str]
    category: str


class EnrichedPersonaSchema(BaseModel):
    """Complete enriched persona schema (original fields + enriched)."""

    # Original fields
    id: int
    type_name: str
    gender: str
    core_feature: str
    health_goals: list[str]
    category: str

    # Enriched fields
    enriched: EnrichedFieldsSchema


class EnrichmentResult(BaseModel):
    """Result of a single persona enrichment."""

    success: bool
    persona_id: int
    enriched_persona: Optional[EnrichedPersonaSchema] = None
    error: Optional[str] = None


class BatchEnrichmentResult(BaseModel):
    """Result of batch persona enrichment."""

    total: int
    success_count: int
    failure_count: int
    results: list[EnrichmentResult]
    failed_ids: list[int]
