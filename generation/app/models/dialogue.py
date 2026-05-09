"""Dialogue database models."""
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base

if TYPE_CHECKING:
    from .persona import ExpandedPersona
    from .event import EventNode


class Dialogue(Base):
    """Dialogue session."""

    __tablename__ = "dialogues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expanded_persona_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("expanded_personas.id"), nullable=False
    )
    current_event_node_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("event_nodes.id")
    )
    context_events: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    knowledge_points: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    expanded_persona: Mapped["ExpandedPersona"] = relationship(
        "ExpandedPersona", back_populates="dialogues"
    )
    current_event: Mapped[Optional["EventNode"]] = relationship("EventNode")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="dialogue", cascade="all, delete-orphan"
    )


class Message(Base):
    """Dialogue message."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dialogue_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("dialogues.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    agent_type: Mapped[Optional[str]] = mapped_column(String(20))
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Relationships
    dialogue: Mapped["Dialogue"] = relationship("Dialogue", back_populates="messages")
