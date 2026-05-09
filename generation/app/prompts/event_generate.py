"""Event type definitions for event graph generation."""

# Event types used for code validation
EVENT_TYPES = {
    "health",
    "life",
    "work",
    "allergy",
    "medication_history",
    "disease_history",
    "medication_preference",
    "diet_preference",
    "lifestyle_economic",
}

# Trap event types (must be included in every generation)
TRAP_EVENT_TYPES = {
    "allergy",
    "medication_history",
    "disease_history",
    "medication_preference",
    "diet_preference",
    "lifestyle_economic",
}
