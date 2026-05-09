#!/usr/bin/env python3
"""Family/friends noise data augmentation CLI.

Type 2 noise: family/friends health condition consultation
- Generate family/friend roles for each persona
- Each role corresponds to multiple health consultation sessions
- Sessions maintain continuity (via context summary)

Usage:
    # Generate noise data for all personas
    python -m augmentation.noise_family.cli generate --dataset-dir /path/to/dataset

    # Generate noise data for the specified persona
    python -m augmentation.noise_family.cli generate --dataset-dir /path/to/dataset --persona-id 1

    # Inject noise data into the dialogue data of the specified persona
    python -m augmentation.noise_family.cli inject --dataset-dir /path/to/dataset --persona-id 1

    # Show help
    python -m augmentation.noise_family.cli --help
    python -m augmentation.noise_family.cli generate --help
    python -m augmentation.noise_family.cli inject --help
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from augmentation.noise_family.config import FamilyNoiseConfig
from augmentation.noise_family.generator import FamilyDialogueGenerator, FamilyNoiseSession, FamilyRole, FamilyNoiseMessage, HealthCondition
from augmentation.noise_family.injector import FamilyNoiseInjector
from app.services.token_tracker import get_token_tracker


async def cmd_generate(args):
    """Generate noise data command."""
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set token tracking stage
    tracker = get_token_tracker()
    tracker.set_stage("noise_family")

    dataset_dir = Path(args.dataset_dir)

    # Get list of personas to process
    if args.persona_id:
        # Specified a single persona
        persona_dirs = [dataset_dir / f"persona_{args.persona_id}"]
        if not persona_dirs[0].exists():
            print(f"❌ Error: persona_{args.persona_id} directory does not exist")
            sys.exit(1)
    else:
        # Process all personas
        persona_dirs = sorted([
            d for d in dataset_dir.iterdir()
            if d.is_dir() and d.name.startswith("persona_")
        ], key=lambda x: int(x.name.split("_")[1]))

    if not persona_dirs:
        print(f"❌ Error: in {dataset_dir} no persona directory found")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("👨‍👩‍👧‍👦 Type 2 noise data generation: family/friends health consultation")
    print("=" * 60)
    print(f"📁 Dataset directory: {dataset_dir}")
    print(f"👥 Number of personas to process: {len(persona_dirs)}")
    print(f"🧑‍🤝‍🧑 Family roles per persona: {args.num_roles}")
    print(f"💬 Sessions per role: {args.sessions_per_role}")
    print(f"🔄 Dialogue turns range: {args.min_turns}-{args.max_turns}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"📏 Max tokens: {args.max_tokens}")
    print("=" * 60)

    if args.dry_run:
        print("\n⚠️  Dry Run mode: only print plan, do not execute")
        for persona_dir in persona_dirs:
            print(f"  - {persona_dir.name}")
        return

    total_sessions = 0
    total_turns = 0

    for persona_dir in persona_dirs:
        persona_id = int(persona_dir.name.split("_")[1])
        background_dir = persona_dir / "background"
        personas_file = background_dir / "generated_personas.json"

        if not personas_file.exists():
            print(f"\n⚠️  Skipped {persona_dir.name}: generated_personas.json not found")
            continue

        print(f"\n📝 Processing {persona_dir.name}...")
        print("-" * 40)

        # Create config - use background directory as data directory
        config = FamilyNoiseConfig(
            data_dir=str(background_dir),
            personas_filename="generated_personas.json",
            num_family_roles=args.num_roles,
            sessions_per_role=args.sessions_per_role,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            model=args.model,
            verbose=not args.quiet,
            dry_run=args.dry_run,
        )

        # Generating noise sessions
        generator = FamilyDialogueGenerator(config)
        noise_sessions = await generator.generate_all()

        # Save noise data to background directory
        output_path = str(background_dir / args.output)
        generator.save_sessions(noise_sessions, output_path)

        # Optional: save family role data
        if args.roles_output:
            roles_output_path = str(background_dir / args.roles_output)
            generator.save_roles(roles_output_path)
            print(f"✓ Family role data saved to: {roles_output_path}")

        total_sessions += len(noise_sessions)
        total_turns += sum(s.turn_count for s in noise_sessions)
        print(f"✓ {persona_dir.name} complete: {len(noise_sessions)} sessions")

    print("\n" + "=" * 60)
    print("✅ Family consultation noise data generation complete")
    print("=" * 60)
    print(f"✓ Total generated sessions: {total_sessions}")
    print(f"✓ Total dialogue turns: {total_turns}")

    # Output token statistics
    tracker.print_console_summary()
    print("=" * 60)


def cmd_inject(args):
    """Inject noise data command."""
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_dir = Path(args.dataset_dir)

    # Get list of personas to process
    if args.persona_id:
        # Specified a single persona
        persona_dirs = [dataset_dir / f"persona_{args.persona_id}"]
        if not persona_dirs[0].exists():
            print(f"❌ Error: persona_{args.persona_id} directory does not exist")
            sys.exit(1)
    else:
        # Process all personas
        persona_dirs = sorted([
            d for d in dataset_dir.iterdir()
            if d.is_dir() and d.name.startswith("persona_")
        ], key=lambda x: int(x.name.split("_")[1]))

    if not persona_dirs:
        print(f"❌ Error: in {dataset_dir} no persona directory found")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📥 Type 2 noise data injection: family/friends health consultation")
    print("=" * 60)
    print(f"📁 Dataset directory: {dataset_dir}")
    print(f"👥 Number of personas to process: {len(persona_dirs)}")
    print(f"📄 Noise data file: {args.noise_file}")
    print(f"📄 Input file: {args.input}")
    print(f"📝 Output file: {args.output}")
    if args.random_seed:
        print(f"🎲 Random seed: {args.random_seed}")
    print("=" * 60)

    total_original = 0
    total_injected = 0
    total_result = 0

    for persona_dir in persona_dirs:
        persona_id = int(persona_dir.name.split("_")[1])
        background_dir = persona_dir / "background"
        eval_dir = persona_dir / "eval"

        # Check if noise data file exists
        noise_file_path = background_dir / args.noise_file
        if not noise_file_path.exists():
            print(f"\n⚠️  Skipped {persona_dir.name}: {args.noise_file} not found")
            continue

        # Check if input dialogue file exists
        input_file_path = eval_dir / args.input
        if not input_file_path.exists():
            print(f"\n⚠️  Skipped {persona_dir.name}: {args.input} not found")
            continue

        print(f"\n📥 Processing {persona_dir.name}...")
        print("-" * 40)

        # Load noise data
        print("  📖 Load noise data...")
        with open(noise_file_path, "r", encoding="utf-8") as f:
            noise_data = json.load(f)

        noise_sessions_data = noise_data.get("noise_sessions", [])
        print(f"    Loaded {len(noise_sessions_data)} noise sessions")

        # Convert to FamilyNoiseSession objects
        noise_sessions = []
        for s in noise_sessions_data:
            messages = [
                FamilyNoiseMessage(
                    turn=m["turn"],
                    role=m["role"],
                    content=m["content"],
                    agent_type=m["agent_type"],
                )
                for m in s["messages"]
            ]
            role_data = s.get("family_role", {})
            # Parse health_conditions
            health_conditions_data = role_data.get("health_conditions", [])
            health_conditions = [
                HealthCondition(
                    condition_name=hc.get("condition_name", ""),
                    icd_category=hc.get("icd_category", ""),
                    severity=hc.get("severity", ""),
                    duration=hc.get("duration", ""),
                    diagnosis_history=hc.get("diagnosis_history", ""),
                    symptoms_detail=hc.get("symptoms_detail", ""),
                    recent_changes=hc.get("recent_changes", ""),
                    lab_results=hc.get("lab_results", ""),
                    medications=hc.get("medications", ""),
                    treatment_history=hc.get("treatment_history", ""),
                    doctor_recommendations=hc.get("doctor_recommendations", ""),
                    concerns=hc.get("concerns", ""),
                    questions_to_ask=hc.get("questions_to_ask", []),
                    upcoming_events=hc.get("upcoming_events", ""),
                    current_status=hc.get("current_status", ""),
                )
                for hc in health_conditions_data
            ]
            role = FamilyRole(
                role_id=role_data.get("role_id", 0),
                persona_id=role_data.get("persona_id", 0),
                relationship=role_data.get("relationship", ""),
                name=role_data.get("name", ""),
                age_range=role_data.get("age_range", ""),
                occupation=role_data.get("occupation", ""),
                personality=role_data.get("personality", ""),
                health_conditions=health_conditions,
            )
            session = FamilyNoiseSession(
                noise_family_id=s["noise_family_id"],
                noise_type=s["noise_type"],
                persona_id=s.get("persona_id", 0),
                family_role=role,
                health_issue=s.get("health_issue", ""),
                turn_count=s["turn_count"],
                messages=messages,
                knowledge_points=s.get("knowledge_points", []),
                session_summary=s.get("session_summary", ""),
                created_at=s["created_at"],
            )
            noise_sessions.append(session)

        # Create config for injection
        config = FamilyNoiseConfig(
            data_dir=str(eval_dir),
            input_filename=args.input,
            output_filename=args.output,
        )

        # Inject noise data
        print("  📥 Inject noise data...")
        injector = FamilyNoiseInjector(config)

        # Load original data
        original_data = injector.load_original_data()

        # Injected noise
        result_data = injector.inject_noise(
            original_data,
            noise_sessions,
            random_seed=args.random_seed,
        )

        # Save result
        output_path = injector.save_data(result_data)

        # Get statistics
        stats = injector.get_injection_stats(result_data)

        total_original += stats['original_sessions']
        total_injected += stats['family_noise_sessions']
        total_result += stats['total_sessions']

        print(f"  ✓ {persona_dir.name} complete:")
        print(f"    Original sessions: {stats['original_sessions']}")
        print(f"    Injected noise: {stats['family_noise_sessions']}")
        print(f"    Total: {stats['total_sessions']}")

    # Output result summary
    print("\n" + "=" * 60)
    print("✅ Family consultation noise data injection complete")
    print("=" * 60)
    print(f"✓ Total original sessions: {total_original}")
    print(f"✓ Total injected noise sessions: {total_injected}")
    print(f"✓ Total output sessions: {total_result}")
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Type 2 noise data (family/friends health consultation) generation and injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # ========== generate Subcommands ==========
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate noise data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples：
    # Generate family consultation noise data for all personas
    python -m augmentation.noise_family.cli generate --dataset-dir /path/to/dataset

    # Generate noise data for the specified persona
    python -m augmentation.noise_family.cli generate --dataset-dir /path/to/dataset --persona-id 1

    # Use custom parameters
    python -m augmentation.noise_family.cli generate --dataset-dir /path/to/dataset --num-roles 3 --sessions-per-role 10
        """,
    )
    gen_parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Dataset directory path (containing persona_X subdirectories)",
    )
    gen_parser.add_argument(
        "--persona-id", type=int, default=None,
        help="Persona ID to process (processes all personas if not specified)",
    )
    gen_parser.add_argument(
        "--output", "-o", type=str, default="noise_family_sessions.json",
        help="Output noise data filename (default: noise_family_sessions.json)",
    )
    gen_parser.add_argument(
        "--roles-output", type=str, default=None,
        help="Filename to separately save family role data (optional)",
    )
    gen_parser.add_argument(
        "--num-roles", type=int, default=5,
        help="Number of family roles to generate per persona (default: 5)",
    )
    gen_parser.add_argument(
        "--sessions-per-role", type=int, default=20,
        help="Number of sessions per role (default: 20)",
    )
    gen_parser.add_argument(
        "--min-turns", type=int, default=5,
        help="Minimum turns per session (default: 5)",
    )
    gen_parser.add_argument(
        "--max-turns", type=int, default=8,
        help="Maximum turns per session (default: 8)",
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="LLM generation temperature (default: 1.0)",
    )
    gen_parser.add_argument(
        "--max-tokens", type=int, default=10000,
        help="LLM max tokens (default: 10000)",
    )
    gen_parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model to use (default: LLM_MODEL from .env)",
    )
    gen_parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print plan, do not execute",
    )
    gen_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce log output",
    )

    # ========== inject Subcommands ==========
    inj_parser = subparsers.add_parser(
        "inject",
        help="Inject noise data into dialogue data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples：
    # Inject noise data for all personas
    python -m augmentation.noise_family.cli inject --dataset-dir /path/to/dataset

    # Inject noise data for the specified persona
    python -m augmentation.noise_family.cli inject --dataset-dir /path/to/dataset --persona-id 1

    # Use random seed
    python -m augmentation.noise_family.cli inject --dataset-dir /path/to/dataset --random-seed 42
        """,
    )
    inj_parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Dataset directory path (containing persona_X subdirectories)",
    )
    inj_parser.add_argument(
        "--persona-id", type=int, default=None,
        help="Persona ID to process (processes all personas if not specified)",
    )
    inj_parser.add_argument(
        "--noise-file", "-n", type=str, default="noise_family_sessions.json",
        help="Noise data filename (default: noise_family_sessions.json)",
    )
    inj_parser.add_argument(
        "--input", "-i", type=str, default="generated_dialogues.json",
        help="Input dialogue data filename (default: generated_dialogues.json)",
    )
    inj_parser.add_argument(
        "--output", "-o", type=str, default="generated_dialogues_with_noise.json",
        help="Output filename for injected data (default: generated_dialogues_with_noise.json)",
    )
    inj_parser.add_argument(
        "--random-seed", type=int, default=None,
        help="Random seed (for reproducible injection positions)",
    )
    inj_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Reduce log output",
    )

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
