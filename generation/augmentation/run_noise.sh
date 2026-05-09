#!/bin/bash

set -e
cd "$(dirname "$0")/.."

DATASET_DIR="${DATASET_DIR:-dataset}"

PERSONA_ID="${PERSONA_ID:-1}"

HEALTH_NUM_SESSIONS="${HEALTH_NUM_SESSIONS:-100}"
HEALTH_MIN_TURNS="${HEALTH_MIN_TURNS:-5}"
HEALTH_MAX_TURNS="${HEALTH_MAX_TURNS:-8}"
HEALTH_TEMPERATURE="${HEALTH_TEMPERATURE:-1.0}"
HEALTH_MAX_TOKENS="${HEALTH_MAX_TOKENS:-10000}"
HEALTH_MODEL="${HEALTH_MODEL:-}"

FAMILY_NUM_ROLES="${FAMILY_NUM_ROLES:-5}"
FAMILY_SESSIONS_PER_ROLE="${FAMILY_SESSIONS_PER_ROLE:-20}"
FAMILY_MIN_TURNS="${FAMILY_MIN_TURNS:-5}"
FAMILY_MAX_TURNS="${FAMILY_MAX_TURNS:-8}"
FAMILY_TEMPERATURE="${FAMILY_TEMPERATURE:-1.0}"
FAMILY_MAX_TOKENS="${FAMILY_MAX_TOKENS:-10000}"
FAMILY_MODEL="${FAMILY_MODEL:-}"

INPUT_FILENAME="${INPUT_FILENAME:-generated_dialogues.json}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-generated_dialogues_with_noise.json}"
RANDOM_SEED="${RANDOM_SEED:-}"

generate_health_noise() {
    echo "Generating type-1 noise (health knowledge chat)..."

    if [ -n "$PERSONA_ID" ]; then
        PERSONA_DIRS=("$DATASET_DIR/persona_$PERSONA_ID")
    else
        PERSONA_DIRS=($(ls -d "$DATASET_DIR"/persona_* 2>/dev/null | sort -V))
    fi

    for PERSONA_DIR in "${PERSONA_DIRS[@]}"; do
        [ ! -d "$PERSONA_DIR" ] && continue
        BACKGROUND_DIR="$PERSONA_DIR/background"

        GEN_ARGS=(
            "--data-dir" "$BACKGROUND_DIR"
            "--output" "noise_health_sessions.json"
            "--num-sessions" "$HEALTH_NUM_SESSIONS"
            "--min-turns" "$HEALTH_MIN_TURNS"
            "--max-turns" "$HEALTH_MAX_TURNS"
            "--temperature" "$HEALTH_TEMPERATURE"
            "--max-tokens" "$HEALTH_MAX_TOKENS"
        )
        [ -n "$HEALTH_MODEL" ] && GEN_ARGS+=("--model" "$HEALTH_MODEL")

        python -m augmentation.noise.cli generate "${GEN_ARGS[@]}"
    done
}

inject_health_noise() {
    echo "Injecting type-1 noise..."

    if [ -n "$PERSONA_ID" ]; then
        PERSONA_DIRS=("$DATASET_DIR/persona_$PERSONA_ID")
    else
        PERSONA_DIRS=($(ls -d "$DATASET_DIR"/persona_* 2>/dev/null | sort -V))
    fi

    for PERSONA_DIR in "${PERSONA_DIRS[@]}"; do
        [ ! -d "$PERSONA_DIR" ] && continue
        BACKGROUND_DIR="$PERSONA_DIR/background"
        EVAL_DIR="$PERSONA_DIR/eval"
        NOISE_FILE="$BACKGROUND_DIR/noise_health_sessions.json"

        [ ! -f "$NOISE_FILE" ] && continue
        [ ! -f "$EVAL_DIR/$INPUT_FILENAME" ] && continue

        INJ_ARGS=(
            "--data-dir" "$EVAL_DIR"
            "--noise-file" "$NOISE_FILE"
            "--input" "$INPUT_FILENAME"
            "--output" "$OUTPUT_FILENAME"
        )
        [ -n "$RANDOM_SEED" ] && INJ_ARGS+=("--random-seed" "$RANDOM_SEED")

        python -m augmentation.noise.cli inject "${INJ_ARGS[@]}"
    done
}

generate_family_noise() {
    echo "Generating type-2 noise (family health consultation)..."

    GEN_ARGS=(
        "--dataset-dir" "$DATASET_DIR"
        "--output" "noise_family_sessions.json"
        "--num-roles" "$FAMILY_NUM_ROLES"
        "--sessions-per-role" "$FAMILY_SESSIONS_PER_ROLE"
        "--min-turns" "$FAMILY_MIN_TURNS"
        "--max-turns" "$FAMILY_MAX_TURNS"
        "--temperature" "$FAMILY_TEMPERATURE"
        "--max-tokens" "$FAMILY_MAX_TOKENS"
    )
    [ -n "$PERSONA_ID" ] && GEN_ARGS+=("--persona-id" "$PERSONA_ID")
    [ -n "$FAMILY_MODEL" ] && GEN_ARGS+=("--model" "$FAMILY_MODEL")

    python -m augmentation.noise_family.cli generate "${GEN_ARGS[@]}"
}

inject_family_noise() {
    echo "Injecting type-2 noise..."

    INJ_ARGS=(
        "--dataset-dir" "$DATASET_DIR"
        "--noise-file" "noise_family_sessions.json"
        "--input" "$OUTPUT_FILENAME"
        "--output" "$OUTPUT_FILENAME"
    )
    [ -n "$PERSONA_ID" ] && INJ_ARGS+=("--persona-id" "$PERSONA_ID")
    [ -n "$RANDOM_SEED" ] && INJ_ARGS+=("--random-seed" "$RANDOM_SEED")

    python -m augmentation.noise_family.cli inject "${INJ_ARGS[@]}"
}

case "${1:-}" in
    generate-health)  generate_health_noise ;;
    generate-family)  generate_family_noise ;;
    inject-health)    inject_health_noise ;;
    inject-family)    inject_family_noise ;;
    generate)         generate_health_noise; generate_family_noise ;;
    inject)           inject_health_noise; inject_family_noise ;;
    all)              generate_health_noise; generate_family_noise; inject_health_noise; inject_family_noise ;;
    *)
        echo "Usage: $0 <command>"
        echo "Commands: generate-health | generate-family | inject-health | inject-family | generate | inject | all"
        echo "Env vars: DATASET_DIR, PERSONA_ID, HEALTH_*, FAMILY_*, INPUT_FILENAME, OUTPUT_FILENAME, RANDOM_SEED"
        exit 1
        ;;
esac

echo "Done."
