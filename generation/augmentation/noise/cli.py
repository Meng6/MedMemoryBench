#!/usr/bin/env python3
"""Type-1 noise data augmentation CLI.

Generates health knowledge chat noise sessions and injects them into dialogue data.
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from augmentation.noise.config import NoiseConfig
from augmentation.noise.generator import NoiseDialogueGenerator, NoiseSession
from augmentation.noise.injector import NoiseDataInjector
from app.services.token_tracker import get_token_tracker


async def cmd_generate(args):
    """Generate noise data."""
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tracker = get_token_tracker()
    tracker.set_stage("noise_health")

    config = NoiseConfig(
        data_dir=args.data_dir,
        num_noise_sessions=args.num_sessions,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model=args.model,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("Type-1 Noise Generation: Health Knowledge Chat")
    print("=" * 60)
    print(f"  Data directory: {config.data_dir}")
    print(f"  Output file: {args.output}")
    print(f"  Sessions: {config.num_noise_sessions}")
    print(f"  Turns range: {config.min_turns}-{config.max_turns}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max tokens: {config.max_tokens}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry Run] Plan printed, not executing.")
        return

    print("\nGenerating noise sessions...")
    print("-" * 40)

    generator = NoiseDialogueGenerator(config)
    noise_sessions = await generator.generate_all()

    output_path = str(Path(config.data_dir) / args.output)
    generator.save_sessions(noise_sessions, output_path)

    print("\n" + "=" * 60)
    print("Noise data generation complete")
    print("=" * 60)
    print(f"  Generated sessions: {len(noise_sessions)}")
    print(f"  Total turns: {sum(s.turn_count for s in noise_sessions)}")
    print(f"  Output: {output_path}")

    tracker.print_console_summary()
    print("=" * 60)


def cmd_inject(args):
    """Inject noise data into dialogue data."""
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = NoiseConfig(
        data_dir=args.data_dir,
        input_filename=args.input,
        output_filename=args.output,
    )

    noise_file_path = Path(args.noise_file)
    if not noise_file_path.exists():
        noise_file_path = Path(config.data_dir) / args.noise_file

    print("\n" + "=" * 60)
    print("Type-1 Noise Injection: Health Knowledge Chat")
    print("=" * 60)
    print(f"  Data directory: {config.data_dir}")
    print(f"  Noise file: {noise_file_path}")
    print(f"  Input file: {args.input}")
    print(f"  Output file: {args.output}")
    if args.random_seed:
        print(f"  Random seed: {args.random_seed}")
    print("=" * 60)

    print("\nLoading noise data...")
    with open(noise_file_path, "r", encoding="utf-8") as f:
        noise_data = json.load(f)

    noise_sessions_data = noise_data.get("noise_sessions", [])
    print(f"  Loaded {len(noise_sessions_data)} noise sessions")

    from augmentation.noise.generator import NoiseSession, NoiseMessage
    noise_sessions = []
    for s in noise_sessions_data:
        messages = [
            NoiseMessage(
                turn=m["turn"],
                role=m["role"],
                content=m["content"],
                agent_type=m["agent_type"],
            )
            for m in s["messages"]
        ]
        session = NoiseSession(
            noise_id=s["noise_id"],
            noise_type=s["noise_type"],
            topic=s["topic"],
            turn_count=s["turn_count"],
            messages=messages,
            knowledge_points=s.get("knowledge_points", []),
            created_at=s["created_at"],
        )
        noise_sessions.append(session)

    print("\nInjecting noise data...")
    injector = NoiseDataInjector(config)

    original_data = injector.load_original_data()

    result_data = injector.inject_noise(
        original_data,
        noise_sessions,
        random_seed=args.random_seed,
    )

    output_path = injector.save_data(result_data)

    stats = injector.get_injection_stats(result_data)

    print("\n" + "=" * 60)
    print("Noise injection complete")
    print("=" * 60)
    print(f"  Original sessions: {stats['original_sessions']}")
    print(f"  Injected noise: {stats['noise_sessions']}")
    if stats.get('other_noise_sessions', 0) > 0:
        print(f"  Other noise sessions: {stats['other_noise_sessions']}")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Noise ratio: {stats['noise_ratio']:.1%}")
    print(f"  Avg noise gap: {stats['avg_gap_between_noise']:.1f} sessions")
    print(f"  Output: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Type-1 noise data (health knowledge chat) generation and injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate noise data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m augmentation.noise.cli generate --num-sessions 100 --output noise_health.json
    python -m augmentation.noise.cli generate --num-sessions 50 --min-turns 5 --max-turns 10
        """,
    )
    gen_parser.add_argument("--data-dir", type=str, default="data",
        help="Data directory path (default: data)")
    gen_parser.add_argument("--output", "-o", type=str, default="noise_health_sessions.json",
        help="Output noise data filename (default: noise_health_sessions.json)")
    gen_parser.add_argument("--num-sessions", "-n", type=int, default=100,
        help="Number of noise sessions (default: 100)")
    gen_parser.add_argument("--min-turns", type=int, default=5,
        help="Minimum turns per session (default: 5)")
    gen_parser.add_argument("--max-turns", type=int, default=8,
        help="Maximum turns per session (default: 8)")
    gen_parser.add_argument("--temperature", type=float, default=1.0,
        help="LLM temperature (default: 1.0)")
    gen_parser.add_argument("--max-tokens", type=int, default=10000,
        help="LLM max tokens (default: 10000)")
    gen_parser.add_argument("--model", type=str, default=None,
        help="LLM model (default: from .env)")
    gen_parser.add_argument("--dry-run", action="store_true",
        help="Only print plan, do not execute")
    gen_parser.add_argument("--quiet", "-q", action="store_true",
        help="Reduce log output")

    inj_parser = subparsers.add_parser(
        "inject",
        help="Inject noise data into dialogue data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m augmentation.noise.cli inject --noise-file noise_health.json --input dialogues.json
    python -m augmentation.noise.cli inject --noise-file noise_health.json --random-seed 42
        """,
    )
    inj_parser.add_argument("--data-dir", type=str, default="data",
        help="Data directory path (default: data)")
    inj_parser.add_argument("--noise-file", "-n", type=str, required=True,
        help="Noise data filename (required)")
    inj_parser.add_argument("--input", "-i", type=str, default="generated_dialogues.json",
        help="Input dialogue data filename (default: generated_dialogues.json)")
    inj_parser.add_argument("--output", "-o", type=str, default="generated_dialogues_with_noise.json",
        help="Output filename (default: generated_dialogues_with_noise.json)")
    inj_parser.add_argument("--random-seed", type=int, default=None,
        help="Random seed for reproducible injection")
    inj_parser.add_argument("--quiet", "-q", action="store_true",
        help="Reduce log output")

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(cmd_generate(args))
    elif args.command == "inject":
        cmd_inject(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
