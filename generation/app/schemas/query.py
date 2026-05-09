"""Query & Answer generation schemas."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Query type enum."""

    EEM = "entity_exact_match"
    TLA = "temporal_localization"
    SUA = "state_update"
    MQ = "multiple_choice"
    IG = "inference_generation"
    MCD = "multi_hop_clinical_deduction"


class Answer(BaseModel):
    """Answer model."""

    content: str = Field(..., description="Answer content")
    is_correct: bool = Field(True, description="Whether this is a correct answer (for multiple choice)")
    explanation: Optional[str] = Field(None, description="Optional explanation")


class SourceKeyPoint(BaseModel):
    """Source knowledge point used for query generation."""

    category: str = Field("", description="Category")
    name: str = Field(..., description="Knowledge point name")
    content: str = Field(..., description="Knowledge point content")
    trap_score: float = Field(0.5, ge=0.0, le=1.0, description="Difficulty score")
    time: Optional[str] = Field(None, description="Event time")
    session_id: int = Field(..., description="Source session ID")


class ReasoningNode(BaseModel):
    """MCD reasoning chain node."""

    node_id: int = Field(..., description="Node ID")
    session_id: int = Field(..., description="Source session ID")
    content: str = Field(..., description="Node content description")
    role: str = Field(..., description="Node role (start/intermediate/end)")
    source_info: Optional[str] = Field(None, description="Source info description")


class MCDMetadata(BaseModel):
    """MCD type metadata."""

    reasoning_chain: list[ReasoningNode] = Field(
        default_factory=list,
        description="Reasoning chain nodes"
    )
    required_memory_nodes: list[str] = Field(
        default_factory=list,
        description="Required memory node descriptions for recall"
    )
    hop_count: int = Field(2, ge=2, le=4, description="Reasoning hop count")
    reasoning_pattern: str = Field("", description="Reasoning pattern type")
    difficulty: str = Field("hard", description="Difficulty level")


class Query(BaseModel):
    """Query model."""

    query_id: str = Field(..., description="Unique query ID")
    session_id: int = Field(..., description="Source session ID")
    query_type: QueryType = Field(..., description="Query type")
    question: str = Field(..., description="Question content")
    answers: list[Answer] = Field(..., description="Answer list")
    source_key_points: list[SourceKeyPoint] = Field(
        default_factory=list,
        description="Source knowledge points for this query"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (e.g. difficulty, entity types)"
    )


class QueryGenerationRequest(BaseModel):
    """Query generation request."""

    input_file: str = Field(
        "data/generated_dialogues.json",
        description="Input dialogue data file path"
    )
    output_file: str = Field(
        "data/generated_queries.json",
        description="Output query data file path"
    )
    queries_per_session: int = Field(
        6,
        ge=1,
        le=50,
        description="Number of queries per session"
    )
    num_eem: int = Field(1, ge=0, le=10, description="EEM type count")
    num_tla: int = Field(1, ge=0, le=10, description="TLA type count")
    num_sua: int = Field(1, ge=0, le=10, description="SUA type count")
    num_mq: int = Field(1, ge=0, le=10, description="MQ type count")
    num_ig: int = Field(1, ge=0, le=10, description="IG type count")
    num_mcd: int = Field(0, ge=0, le=5, description="MCD type count (generated every N sessions)")

    generate_every_n_sessions: int = Field(
        5,
        ge=1,
        description="Generate queries (EEM/TLA/SUA/MQ/IG) every N sessions"
    )
    mcd_generate_every_n_sessions: int = Field(
        10,
        ge=5,
        description="Generate MCD queries every N sessions (requires more context)"
    )
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    max_tokens: int = Field(
        2000,
        description="LLM max generation tokens"
    )


class QueryGenerationResponse(BaseModel):
    """Query generation response."""

    total_sessions: int = Field(..., description="Total session count")
    processed_sessions: int = Field(..., description="Processed session count")
    skipped_sessions: int = Field(..., description="Skipped session count")
    total_queries: int = Field(..., description="Total generated query count")
    queries_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Query count by type"
    )
    output_file: str = Field(..., description="Output file path")
    errors: list[dict] = Field(
        default_factory=list,
        description="Error list"
    )
