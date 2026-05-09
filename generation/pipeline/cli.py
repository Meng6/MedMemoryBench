"""Command-line interface."""
import argparse
import asyncio
import logging
import sys
from typing import Optional

from app.database import close_db
from app.services.token_tracker import get_token_tracker
from .config import (
    GenerationConfig,
    PersonaConfig,
    EventConfig,
    DialogueConfig,
    QueryConfig,
    LLMConfig,
    QUICK_TEST_CONFIG,
    PRODUCTION_CONFIG,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from .generator import DataGenerator
from .query_generator import QueryGenerator
from app.schemas.query import QueryGenerationRequest


def setup_logging(verbose: bool = False):
    """Configure logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_ids(ids_str: Optional[str]) -> Optional[list[int]]:
    """Parse a comma-separated ID list string into a list of integers."""
    if not ids_str:
        return None
    try:
        return [int(x.strip()) for x in ids_str.split(",")]
    except ValueError:
        raise ValueError(f"Invalid ID list format: {ids_str}")


def add_llm_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """Add LLM hyperparameter options to a parser."""
    parser.add_argument(
        f"--{prefix}temperature" if prefix else "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"LLM temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        f"--{prefix}max-tokens" if prefix else "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"LLM max generation tokens (default: {DEFAULT_MAX_TOKENS})",
    )


def get_llm_config_from_args(args, prefix: str = "") -> LLMConfig:
    """Build LLMConfig from command-line arguments."""
    temp_attr = f"{prefix}temperature" if prefix else "temperature"
    tokens_attr = f"{prefix}max_tokens" if prefix else "max_tokens"

    temperature = getattr(args, temp_attr.replace("-", "_"), DEFAULT_TEMPERATURE)
    max_tokens = getattr(args, tokens_attr.replace("-", "_"), DEFAULT_MAX_TOKENS)

    return LLMConfig(temperature=temperature, max_tokens=max_tokens)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Data generation pipeline - batch generate personas, event graphs, and dialogues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Generate personas
  python -m pipeline.cli generate-personas --count 10 --output data/personas.json

  # Generate event graphs for specific personas
  python -m pipeline.cli generate-events --persona-ids "1,2,3" --output data/events.json

  # Generate dialogue sessions (10 personas x 5 sessions = 50 sessions)
  python -m pipeline.cli generate-dialogues --persona-ids "1,2,3,4,5,6,7,8,9,10" --sessions 5 --turns 10

  # Quick test with preset config
  python -m pipeline.cli run-all --preset quick

  # Production run with preset config
  python -m pipeline.cli run-all --preset production
        """,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose logs",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # import-personas subcommand
    import_parser = subparsers.add_parser(
        "import-personas",
        help="Import base personas into database",
        description="Import base persona data from a JSON file",
    )
    import_parser.add_argument(
        "--input",
        type=str,
        default="../user_personas.json",
        help="Input file path (default: ../user_personas.json)",
    )

    # generate-personas subcommand
    personas_parser = subparsers.add_parser(
        "generate-personas",
        help="Generate user personas",
        description="Batch expand base personas into detailed user background profiles",
    )
    personas_parser.add_argument(
        "--count",
        type=int,
        help="Number to generate (if persona-ids not specified)",
    )
    personas_parser.add_argument(
        "--persona-ids",
        type=str,
        help='Base persona IDs to expand (comma-separated, e.g. "1,2,3")',
    )
    personas_parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip already-expanded personas (default: enabled)",
    )
    personas_parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip already-expanded personas",
    )
    personas_parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrency level (default: 5)",
    )
    personas_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_personas.json",
        help="Output file path (default: data/generated_personas.json)",
    )
    add_llm_args(personas_parser)

    # generate-events subcommand
    events_parser = subparsers.add_parser(
        "generate-events",
        help="Generate event graphs",
        description="Generate health event timelines for user personas",
    )
    events_parser.add_argument(
        "--persona-ids",
        type=str,
        help='Persona IDs (comma-separated, e.g. "1,2,3")',
    )
    events_parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip personas with existing event graphs (default: enabled)",
    )
    events_parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip personas with existing event graphs",
    )
    events_parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Event start date YYYY-MM-DD (default: 2024-01-01)",
    )
    events_parser.add_argument(
        "--time-span",
        type=int,
        default=365,
        help="Event time span in days (default: 365)",
    )
    events_parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Concurrency level (default: 3)",
    )
    events_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_events.json",
        help="Output file path (default: data/generated_events.json)",
    )
    events_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Events per batch (default: 10)",
    )
    events_parser.add_argument(
        "--max-total-events",
        type=int,
        default=100,
        help="Maximum total events (default: 100)",
    )
    add_llm_args(events_parser)

    # generate-trap-events subcommand
    trap_events_parser = subparsers.add_parser(
        "generate-trap-events",
        help="Generate trap events (phase 1 of two-phase mode)",
        description="Generate 6 types of fixed trap events for user personas",
    )
    trap_events_parser.add_argument(
        "--persona-ids",
        type=str,
        help='Persona IDs (comma-separated, e.g. "1,2,3")',
    )
    trap_events_parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Event date YYYY-MM-DD (default: 2024-01-01)",
    )
    trap_events_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_trap_events.json",
        help="Output file path (default: data/generated_trap_events.json)",
    )
    add_llm_args(trap_events_parser)

    # generate-regular-events subcommand
    regular_events_parser = subparsers.add_parser(
        "generate-regular-events",
        help="Generate regular events (phase 2 of two-phase mode)",
        description="Generate regular events in phases based on clinical reports, inserting trap events into phase 1",
    )
    regular_events_parser.add_argument(
        "--persona-ids",
        type=str,
        help='Persona IDs (comma-separated, e.g. "1,2,3")',
    )
    regular_events_parser.add_argument(
        "--trap-events-input",
        type=str,
        default="data/generated_trap_events.json",
        help="Trap events input file (default: data/generated_trap_events.json)",
    )
    regular_events_parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip personas with existing event graphs (default: enabled)",
    )
    regular_events_parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip personas with existing event graphs",
    )
    regular_events_parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Event start date YYYY-MM-DD (default: 2024-01-01)",
    )
    regular_events_parser.add_argument(
        "--time-span",
        type=int,
        default=365,
        help="Event time span in days (default: 365)",
    )
    regular_events_parser.add_argument(
        "--events-per-phase",
        type=int,
        default=20,
        help="Events per phase (default: 20, 5 phases = 100 total)",
    )
    regular_events_parser.add_argument(
        "--max-total-events",
        type=int,
        default=100,
        help="Maximum total events (default: 100)",
    )
    regular_events_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_events.json",
        help="Output file path (default: data/generated_events.json)",
    )
    add_llm_args(regular_events_parser)

    # generate-dialogues subcommand
    dialogues_parser = subparsers.add_parser(
        "generate-dialogues",
        help="Generate dialogue sessions",
        description="Generate doctor-patient dialogue interactions (saved by sessions)",
    )
    dialogues_parser.add_argument(
        "--persona-ids",
        type=str,
        help='Persona IDs (comma-separated, e.g. "1,2,3")',
    )
    dialogues_parser.add_argument(
        "--sessions",
        type=int,
        default=5,
        help="Sessions per persona (default: 5)",
    )
    dialogues_parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Max turns per dialogue (default: 10)",
    )
    dialogues_parser.add_argument(
        "--no-natural-end",
        dest="allow_natural_end",
        action="store_false",
        default=True,
        help="Disable natural ending (dialogue must reach max turns)",
    )
    dialogues_parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip personas with sufficient existing dialogues",
    )
    dialogues_parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrency level (default: 2)",
    )
    dialogues_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_dialogues.json",
        help="Output file path (default: data/generated_dialogues.json)",
    )
    dialogues_parser.add_argument(
        "--no-verbose",
        dest="export_verbose",
        action="store_false",
        default=True,
        help="Exclude verbose info (persona, events) from export",
    )
    add_llm_args(dialogues_parser)

    # run-all subcommand
    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run full pipeline (personas -> events -> dialogues)",
        description="Run all three generation phases sequentially",
    )
    run_all_parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "production"],
        help="Use preset config (quick: fast test, production: full run)",
    )

    # generate-queries subcommand
    queries_parser = subparsers.add_parser(
        "generate-queries",
        help="Generate evaluation queries",
        description="Generate Memory Benchmark queries & answers from dialogue data using accumulated knowledge points",
    )
    queries_parser.add_argument(
        "--input",
        type=str,
        default="data/generated_dialogues.json",
        help="Input dialogue data file (default: data/generated_dialogues.json)",
    )
    queries_parser.add_argument(
        "--output",
        type=str,
        default="data/generated_queries.json",
        help="Output query data file (default: data/generated_queries.json)",
    )
    queries_parser.add_argument(
        "--queries-per-session",
        type=int,
        default=6,
        help="Queries per session (default: 6)",
    )
    queries_parser.add_argument(
        "--num-eem",
        type=int,
        default=1,
        help="EEM (Entity Exact Match) count (default: 1)",
    )
    queries_parser.add_argument(
        "--num-tla",
        type=int,
        default=1,
        help="TLA (Temporal Localization Accuracy) count (default: 1)",
    )
    queries_parser.add_argument(
        "--num-sua",
        type=int,
        default=1,
        help="SUA (State Update Accuracy) count (default: 1)",
    )
    queries_parser.add_argument(
        "--num-mq",
        type=int,
        default=1,
        help="MQ (Multiple-choice Question) count (default: 1)",
    )
    queries_parser.add_argument(
        "--num-ig",
        type=int,
        default=1,
        help="IG (Inference Generation) count (default: 1)",
    )
    queries_parser.add_argument(
        "--num-mcd",
        type=int,
        default=1,
        help="MCD (Multi-hop Clinical Deduction) count (default: 1, generated every 10 sessions)",
    )
    queries_parser.add_argument(
        "--generate-every",
        type=int,
        default=5,
        help="Generate queries every N sessions (default: 5)",
    )
    queries_parser.add_argument(
        "--mcd-generate-every",
        type=int,
        default=10,
        help="Generate MCD queries every N sessions (default: 10)",
    )
    add_llm_args(queries_parser)

    return parser


async def cmd_import_personas(args):
    """Execute base persona import command."""
    import json
    from pathlib import Path
    from app.database import AsyncSessionLocal, init_db
    from app.services.persona import import_base_personas

    setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found - {input_path}")
        return

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("[Pipeline] Initializing database...")
    logger.info("=" * 80)

    await init_db()

    with open(input_path, "r", encoding="utf-8") as f:
        personas_data = json.load(f)

    async with AsyncSessionLocal() as db:
        count = await import_base_personas(db, personas_data)
        await db.commit()

    print("\n" + "=" * 80)
    print("Import complete!")
    print("=" * 80)
    print(f"File: {input_path}")
    print(f"Total records: {len(personas_data)}")
    print(f"Newly imported: {count}")
    print(f"Already existed: {len(personas_data) - count}")
    print("=" * 80)

    await close_db()


async def cmd_generate_personas(args):
    """Execute persona generation command."""
    setup_logging(args.verbose)

    tracker = get_token_tracker()
    tracker.set_stage("persona")

    config = PersonaConfig(
        count=args.count,
        persona_ids=parse_ids(args.persona_ids),
        skip_existing=args.skip_existing,
        concurrency=args.concurrency,
        export_path=args.output,
        llm=get_llm_config_from_args(args),
    )

    generator = DataGenerator(GenerationConfig(persona=config))
    await generator.initialize()
    result = await generator.generate_personas(config)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Generated: {result['generated']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result['errors']}")
    if result.get('export'):
        print(f"Export path: {result['export']['path']}")

    tracker.print_console_summary()
    print("=" * 80)

    await close_db()


async def cmd_generate_events(args):
    """Execute event generation command."""
    setup_logging(args.verbose)

    tracker = get_token_tracker()
    tracker.set_stage("event")

    config = EventConfig(
        persona_ids=parse_ids(args.persona_ids),
        skip_existing=args.skip_existing,
        start_date=args.start_date,
        time_span_days=args.time_span,
        concurrency=args.concurrency,
        export_path=args.output,
        llm=get_llm_config_from_args(args),
        events_per_phase=args.batch_size,
        max_total_events=args.max_total_events,
    )

    generator = DataGenerator(GenerationConfig(event=config))
    await generator.initialize()
    result = await generator.generate_events(config)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Mode: incremental (batch_size={config.batch_size}, max={config.max_total_events})")
    print(f"Generated: {result['generated']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result['errors']}")
    if result.get('export'):
        print(f"Total events: {result['export'].get('total_events', 0)}")
        print(f"Export path: {result['export']['path']}")

    tracker.print_console_summary()
    print("=" * 80)

    await close_db()


async def cmd_generate_trap_events(args):
    """Execute trap event generation command (phase 1 of two-phase mode)."""
    setup_logging(args.verbose)

    tracker = get_token_tracker()
    tracker.set_stage("trap_event")

    config = EventConfig(
        persona_ids=parse_ids(args.persona_ids),
        start_date=args.start_date,
        trap_events_export_path=args.output,
        llm=get_llm_config_from_args(args),
    )

    generator = DataGenerator(GenerationConfig(event=config))
    await generator.initialize()
    result = await generator.generate_trap_events(config)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("Mode: trap event generation (two-phase mode, phase 1)")
    print(f"Generated: {result['generated']}")
    print(f"Errors: {result['errors']}")
    if result.get('export'):
        print(f"Total trap events: {result['export'].get('total_events', 0)}")
        print(f"Export path: {result['export']['path']}")

    tracker.print_console_summary()
    print("=" * 80)

    await close_db()


async def cmd_generate_regular_events(args):
    """Execute regular event generation command (phase 2 of two-phase mode)."""
    setup_logging(args.verbose)

    tracker = get_token_tracker()
    tracker.set_stage("regular_event")

    config = EventConfig(
        persona_ids=parse_ids(args.persona_ids),
        skip_existing=args.skip_existing,
        start_date=args.start_date,
        time_span_days=args.time_span,
        trap_events_export_path=args.trap_events_input,
        export_path=args.output,
        llm=get_llm_config_from_args(args),
        events_per_phase=args.events_per_phase,
        max_total_events=args.max_total_events,
    )

    generator = DataGenerator(GenerationConfig(event=config))
    await generator.initialize()
    result = await generator.generate_regular_events(config)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print("Mode: regular event generation (two-phase mode, phase 2, clinical-report-guided)")
    print(f"Generated: {result['generated']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result['errors']}")
    if result.get('export'):
        print(f"Total events: {result['export'].get('total_events', 0)}")
        print(f"Export path: {result['export']['path']}")

    tracker.print_console_summary()
    print("=" * 80)

    await close_db()


async def cmd_generate_dialogues(args):
    """Execute dialogue generation command."""
    setup_logging(args.verbose)

    tracker = get_token_tracker()
    tracker.set_stage("dialogue")

    config = DialogueConfig(
        persona_ids=parse_ids(args.persona_ids),
        sessions_per_persona=args.sessions,
        max_turns=args.turns,
        allow_natural_end=args.allow_natural_end,
        skip_existing=args.skip_existing,
        concurrency=args.concurrency,
        export_path=args.output,
        export_verbose=args.export_verbose,
        llm=get_llm_config_from_args(args),
    )

    generator = DataGenerator(GenerationConfig(dialogue=config))
    await generator.initialize()
    result = await generator.generate_dialogues(config)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"Sessions generated: {result['generated']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Errors: {result['errors']}")
    if result.get('export'):
        print(f"Total turns: {result['export'].get('total_turns', 0)}")
        print(f"Export path: {result['export']['path']}")

    tracker.print_console_summary()
    print("=" * 80)

    await close_db()


async def cmd_run_all(args):
    """Execute the full pipeline."""
    setup_logging(args.verbose)

    if args.preset == "quick":
        config = QUICK_TEST_CONFIG
        print("\nUsing quick test config...")
    elif args.preset == "production":
        config = PRODUCTION_CONFIG
        print("\nUsing production config...")
    else:
        print("Please specify --preset (quick or production)")
        return

    generator = DataGenerator(config)
    await generator.initialize()

    print("\n" + "=" * 80)
    print("Phase 1/3: Generate personas")
    print("=" * 80)
    await generator.generate_personas()

    print("\n" + "=" * 80)
    print("Phase 2/3: Generate event graphs")
    print("=" * 80)
    await generator.generate_events()

    print("\n" + "=" * 80)
    print("Phase 3/3: Generate dialogues")
    print("=" * 80)
    result = await generator.generate_dialogues()

    print("\n" + "=" * 80)
    print("All done!")
    print("=" * 80)

    await close_db()


async def cmd_generate_queries(args):
    """Execute query generation command."""
    setup_logging(args.verbose)

    tracker = get_token_tracker()
    tracker.set_stage("query")

    request = QueryGenerationRequest(
        input_file=args.input,
        output_file=args.output,
        queries_per_session=args.queries_per_session,
        num_eem=args.num_eem,
        num_tla=args.num_tla,
        num_sua=args.num_sua,
        num_mq=args.num_mq,
        num_ig=args.num_ig,
        num_mcd=args.num_mcd,
        generate_every_n_sessions=args.generate_every,
        mcd_generate_every_n_sessions=args.mcd_generate_every,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    generator = QueryGenerator(request)
    result = await generator.generate()

    print("\n" + "=" * 80)
    print("Query generation complete!")
    print("=" * 80)
    print(f"Total sessions: {result.total_sessions}")
    print(f"Processed: {result.processed_sessions}")
    print(f"Skipped: {result.skipped_sessions}")
    print(f"Generated queries: {result.total_queries}")
    print(f"By type: {result.queries_by_type}")
    print(f"Errors: {len(result.errors)}")
    print(f"Output file: {result.output_file}")

    tracker.print_console_summary()
    print("=" * 80)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "import-personas":
        asyncio.run(cmd_import_personas(args))
    elif args.command == "generate-personas":
        asyncio.run(cmd_generate_personas(args))
    elif args.command == "generate-events":
        asyncio.run(cmd_generate_events(args))
    elif args.command == "generate-trap-events":
        asyncio.run(cmd_generate_trap_events(args))
    elif args.command == "generate-regular-events":
        asyncio.run(cmd_generate_regular_events(args))
    elif args.command == "generate-dialogues":
        asyncio.run(cmd_generate_dialogues(args))
    elif args.command == "generate-queries":
        asyncio.run(cmd_generate_queries(args))
    elif args.command == "run-all":
        asyncio.run(cmd_run_all(args))


if __name__ == "__main__":
    main()
