"""Event graph database models."""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .persona import ExpandedPersona


class EventGraph(Base):
    """Event graph."""

    __tablename__ = "event_graphs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expanded_persona_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expanded_personas.id"), nullable=False, unique=True
    )
    start_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    time_span_days: Mapped[int] = mapped_column(Integer, default=90)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    expanded_persona: Mapped["ExpandedPersona"] = relationship(
        "ExpandedPersona", back_populates="event_graph"
    )
    event_nodes: Mapped[list["EventNode"]] = relationship(
        "EventNode", back_populates="event_graph", cascade="all, delete-orphan"
    )


class EventNode(Base):
    """Event node."""

    __tablename__ = "event_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_graph_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("event_graphs.id"), nullable=False
    )
    event: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    event_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    triggered_by: Mapped[Optional[list]] = mapped_column(JSON, default=list)

    # Relationships
    event_graph: Mapped["EventGraph"] = relationship(
        "EventGraph", back_populates="event_nodes"
    )
