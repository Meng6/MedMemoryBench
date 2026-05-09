"""Persona database models."""
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class BasePersona(Base):
    """Base user persona (imported from user_personas.json)."""

    __tablename__ = "base_personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type_name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    core_feature: Mapped[Optional[str]] = mapped_column(Text)
    health_goals: Mapped[Optional[list]] = mapped_column(JSON)
    category: Mapped[Optional[str]] = mapped_column(String(50))

    # Relationships
    expanded_personas: Mapped[list["ExpandedPersona"]] = relationship(
        "ExpandedPersona", back_populates="base_persona", cascade="all, delete-orphan"
    )


class ExpandedPersona(Base):
    """Expanded user persona with enriched_data JSON field."""

    __tablename__ = "expanded_personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    base_persona_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("base_personas.id"), nullable=False
    )
    enriched_data: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    base_persona: Mapped["BasePersona"] = relationship(
        "BasePersona", back_populates="expanded_personas"
    )
    event_graph: Mapped[Optional["EventGraph"]] = relationship(
        "EventGraph", back_populates="expanded_persona", uselist=False, cascade="all, delete-orphan"
    )
    dialogues: Mapped[list["Dialogue"]] = relationship(
        "Dialogue", back_populates="expanded_persona", cascade="all, delete-orphan"
    )


# Import here to avoid circular imports
from .event import EventGraph
from .dialogue import Dialogue
