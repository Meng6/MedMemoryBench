"""Pipeline usage examples."""
import asyncio
import logging

from pipeline import (
    DataGenerator,
    GenerationConfig,
    PersonaConfig,
    EventConfig,
    DialogueConfig,
    QUICK_TEST_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def example_1_generate_personas():
    """Example 1: Generate user personas."""
    print("\n" + "=" * 80)
    print("Example 1: Generate user personas")
    print("=" * 80)

    config = GenerationConfig(
        persona=PersonaConfig(
            persona_ids=[1, 2, 3, 4, 5],
            concurrency=3,
            export_path="data/example_personas.json",
        )
    )

    generator = DataGenerator(config)
    await generator.initialize()
    result = await generator.generate_personas()

    print(f"\nGenerated {result['generated']} personas")
    print(f"Skipped {result['skipped']} personas")
    print(f"Exported to: {result['export']['path']}")


async def example_2_generate_events():
    """Example 2: Generate event graphs."""
    print("\n" + "=" * 80)
    print("Example 2: Generate event graphs")
    print("=" * 80)

    config = GenerationConfig(
        event=EventConfig(
            persona_ids=[1, 2, 3],
            start_date="2024-01-01",
            time_span_days=120,
            batch_size=10,
            max_total_events=50,
            concurrency=2,
            export_path="data/example_events.json",
        )
    )

    generator = DataGenerator(config)
    await generator.initialize()
    result = await generator.generate_events()

    print(f"\nGenerated {result['generated']} event graphs")
    print(f"Total events: {result['export']['total_events']}")
    print(f"Exported to: {result['export']['path']}")


async def example_3_generate_dialogues():
    """Example 3: Generate dialogue sessions (10 personas x 5 sessions = 50 sessions)."""
    print("\n" + "=" * 80)
    print("Example 3: Generate dialogue sessions")
    print("=" * 80)

    config = GenerationConfig(
        dialogue=DialogueConfig(
            persona_ids=list(range(1, 11)),
            sessions_per_persona=5,
            max_turns=12,
            allow_natural_end=True,
            concurrency=2,
            export_path="data/example_dialogues.json",
        )
    )

    generator = DataGenerator(config)
    await generator.initialize()
    result = await generator.generate_dialogues()

    print(f"\nGenerated {result['generated']} sessions")
    print(f"Total turns: {result['export']['total_turns']}")
    print(f"Avg turns/session: {result['export']['total_turns'] / result['generated']:.1f}")
    print(f"Exported to: {result['export']['path']}")


async def example_4_full_pipeline():
    """Example 4: Run full pipeline."""
    print("\n" + "=" * 80)
    print("Example 4: Full pipeline (personas -> events -> dialogues)")
    print("=" * 80)

    config = GenerationConfig(
        persona=PersonaConfig(
            persona_ids=[1, 2, 3],
            concurrency=2,
            export_path="data/full_personas.json",
        ),
        event=EventConfig(
            persona_ids=[1, 2, 3],
            time_span_days=60,
            batch_size=8,
            max_total_events=30,
            concurrency=2,
            export_path="data/full_events.json",
        ),
        dialogue=DialogueConfig(
            persona_ids=[1, 2, 3],
            sessions_per_persona=3,
            max_turns=8,
            concurrency=1,
            export_path="data/full_dialogues.json",
        ),
    )

    generator = DataGenerator(config)
    await generator.initialize()

    print("\n[Phase 1/3] Generating personas...")
    persona_result = await generator.generate_personas()

    print("\n[Phase 2/3] Generating events...")
    event_result = await generator.generate_events()

    print("\n[Phase 3/3] Generating dialogues...")
    dialogue_result = await generator.generate_dialogues()

    print("\n" + "=" * 80)
    print("All done!")
    print("=" * 80)
    print(f"Personas: {persona_result['generated']}")
    print(f"Event graphs: {event_result['generated']}")
    print(f"Dialogue sessions: {dialogue_result['generated']}")
    print(f"Total turns: {dialogue_result['export']['total_turns']}")


async def example_5_use_preset():
    """Example 5: Use preset config."""
    print("\n" + "=" * 80)
    print("Example 5: Using preset config (QUICK_TEST_CONFIG)")
    print("=" * 80)

    generator = DataGenerator(QUICK_TEST_CONFIG)
    await generator.initialize()

    await generator.generate_personas()
    await generator.generate_events()
    result = await generator.generate_dialogues()

    print(f"\nQuick test done, generated {result['generated']} sessions")


async def example_6_incremental_generation():
    """Example 6: Incremental generation (append data)."""
    print("\n" + "=" * 80)
    print("Example 6: Incremental generation")
    print("=" * 80)

    generator = DataGenerator(GenerationConfig())
    await generator.initialize()

    print("\n[Round 1] Generating 5 sessions...")
    config1 = DialogueConfig(
        persona_ids=[1, 2, 3],
        sessions_per_persona=5,
        skip_existing=False,
        export_path="data/incremental_v1.json",
    )
    result1 = await generator.generate_dialogues(config1)
    print(f"Generated {result1['generated']} sessions")

    print("\n[Round 2] Appending 5 more sessions...")
    config2 = DialogueConfig(
        persona_ids=[1, 2, 3],
        sessions_per_persona=10,
        skip_existing=True,
        export_path="data/incremental_v2.json",
    )
    result2 = await generator.generate_dialogues(config2)
    print(f"Generated {result2['generated']} sessions this round")


async def example_7_batch_processing():
    """Example 7: Batch processing multiple persona groups."""
    print("\n" + "=" * 80)
    print("Example 7: Batch processing")
    print("=" * 80)

    persona_batches = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    generator = DataGenerator(GenerationConfig())
    await generator.initialize()

    total_sessions = 0

    for i, batch in enumerate(persona_batches, 1):
        print(f"\n[Batch {i}/{len(persona_batches)}] Processing personas {batch}...")

        config = DialogueConfig(
            persona_ids=batch,
            sessions_per_persona=3,
            max_turns=8,
            export_path=f"data/batch_{i}.json",
        )

        result = await generator.generate_dialogues(config)
        total_sessions += result["generated"]
        print(f"This batch generated {result['generated']} sessions")

    print(f"\nAll done, total {total_sessions} sessions generated")


async def main():
    """Run examples interactively."""
    print("\n" + "=" * 80)
    print("Pipeline Usage Examples")
    print("=" * 80)
    print("\nSelect an example to run:")
    print("1. Generate user personas")
    print("2. Generate event graphs")
    print("3. Generate dialogue sessions (10 personas x 5 sessions)")
    print("4. Run full pipeline")
    print("5. Use preset config")
    print("6. Incremental generation")
    print("7. Batch processing")
    print("0. Run all examples")

    choice = input("\nEnter option (0-7): ").strip()

    examples = {
        "1": example_1_generate_personas,
        "2": example_2_generate_events,
        "3": example_3_generate_dialogues,
        "4": example_4_full_pipeline,
        "5": example_5_use_preset,
        "6": example_6_incremental_generation,
        "7": example_7_batch_processing,
    }

    if choice == "0":
        for example_func in examples.values():
            await example_func()
    elif choice in examples:
        await examples[choice]()
    else:
        print("Invalid option")


if __name__ == "__main__":
    asyncio.run(main())
