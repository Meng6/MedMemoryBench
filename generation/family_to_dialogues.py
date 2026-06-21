"""Convert family-consultation noise rows from a flat parquet into the
`generated_dialogues.json` format consumed by `pipeline.cli generate-queries`.

Input : ../data/en/dialogues_with_noise.parquet  (one row per dialogue turn, downloaded via https://huggingface.co/datasets/Cyan27/MedMemoryBench)
Filter: noise_type == "family_health_consultation"
Output: data/generated_dialogues.json            ({"metadata":..., "sessions":[...]})

[Note] you need to download en/dialogues_with_noise.parquet from Huggingface first
Run:
    python family_to_dialogues.py \
        --input data/dialogues_with_noise.parquet \
        --output data/generated_dialogues.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


NOISE_TYPE = "family_health_consultation"


def parse_kps(raw):
    """Return a list[dict] from a knowledge_points cell.

    The cell may be a JSON string, a list, or a numpy array depending on how
    parquet stored it. Anything unparseable yields an empty list.
    """
    if raw is None:
        return []
    # pandas may give a numpy array / list of dicts directly
    if isinstance(raw, (list, tuple)):
        return [dict(x) for x in raw]
    try:
        import numpy as np
        if isinstance(raw, np.ndarray):
            return [dict(x) for x in raw.tolist()]
    except Exception:
        pass
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() == "nan":
            return []
        try:
            val = json.loads(s)
            return val if isinstance(val, list) else []
        except json.JSONDecodeError:
            return []
    return []


def clean_event_id(val):
    """Return an int event_id or None (family noise rows have NaN)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return val


def kp_key(kp):
    return (kp.get("category", ""), kp.get("name", ""), kp.get("content", ""))


def build_sessions(df):
    """Group flat turn-rows into accumulated-KP session objects."""
    sessions = []
    # Stable per-persona, per-session ordering; turns ordered within a session.
    for persona_id, pdf in df.groupby("persona_id", sort=True):
        # accumulated, deduplicated KP list across this persona's sessions
        acc_kps = []
        seen = set()

        session_ids = sorted(pdf["session_id"].unique(), key=lambda x: (x is None, x))
        for session_id in session_ids:
            sdf = pdf[pdf["session_id"] == session_id]
            if "turn" in sdf.columns:
                sdf = sdf.sort_values("turn", kind="stable")

            # --- messages / turns ---
            messages = []
            for i, (_, row) in enumerate(sdf.iterrows(), start=1):
                messages.append({
                    "turn": int(row["turn"]) if "turn" in row and pd.notna(row["turn"]) else i,
                    "role": row.get("role", ""),
                    "content": row.get("content", ""),
                    "agent_type": "user" if row.get("role") == "user" else "assistant",
                })

            # --- knowledge points (parse once per session, then accumulate) ---
            first = sdf.iloc[0]
            raw_kps = parse_kps(first.get("knowledge_points"))
            for kp in raw_kps:
                kp = dict(kp)
                # query-gen filters/sorts on session_id; family KPs lack it
                kp.setdefault("session_id", int(session_id))
                # trap_score is used for EEM/TLA selection; default mid if absent
                kp.setdefault("trap_score", 0.5)
                k = kp_key(kp)
                if k not in seen:
                    seen.add(k)
                    acc_kps.append(kp)

            event_info = first.get("event_info")
            if isinstance(event_info, float) and pd.isna(event_info):
                event_info = None

            sessions.append({
                "session_id": int(session_id),
                "persona_id": int(persona_id),
                "event_id": clean_event_id(first.get("event_id")),
                "status": "completed",
                "turn_count": len(messages),
                "messages": messages,
                "turns": messages,  # MCD enrichment reads `turns`
                # accumulated snapshot up to and including this session
                "knowledge_points": [dict(kp) for kp in acc_kps],
                "kp_count": len(acc_kps),
                "noise_type": NOISE_TYPE,
                "event_info": event_info,
            })

    sessions.sort(key=lambda s: (s["persona_id"], s["session_id"]))
    return sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/en/dialogues_with_noise.parquet")
    ap.add_argument("--output", default="data/generated_dialogues.json")
    ap.add_argument("--noise-type", default=NOISE_TYPE)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    fam = df[df["noise_type"] == args.noise_type].copy()
    print(f"Family rows ({args.noise_type}): {len(fam)}")
    if fam.empty:
        raise SystemExit("No family rows found — check --noise-type / column names.")

    sessions = build_sessions(fam)
    total_turns = sum(s["turn_count"] for s in sessions)
    total_kps = sum(len(s["knowledge_points"]) for s in sessions)
    personas = sorted({s["persona_id"] for s in sessions})

    out = {
        "metadata": {
            "export_time": datetime.now().isoformat(),
            "source_file": str(args.input),
            "noise_type": args.noise_type,
            "total_sessions": len(sessions),
            "total_turns": total_turns,
            "total_key_points": total_kps,
            "personas_count": len(personas),
            "personas": personas,
            "key_points_structure": {
                "description": "Accumulated KP list - each session contains all KPs up to that session (additive, deduplicated)",
            },
        },
        "sessions": sessions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(sessions)} sessions for personas {personas} -> {out_path}")
    print(f"  total_turns={total_turns}, total_key_points(last-session-acc summed)={total_kps}")


if __name__ == "__main__":
    main()
