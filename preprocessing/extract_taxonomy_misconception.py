import pandas as pd
import re
from pathlib import Path

# --- Config ---
INPUT_PATH = "data/train_failures.csv"
OUTPUT_PATH = "preprocessing/output/mathdial_draft.csv"
TOPIC_KEYWORDS = ["fraction", "denominator", "numerator","ratio"]  # match your other two sources
SAMPLE_SIZE = 50
RANDOM_SEED = 42

def matches_topic(question_text: str) -> bool:
    text = str(question_text).lower()
    return any(kw in text for kw in TOPIC_KEYWORDS)

def parse_first_turns(conversation: str, max_turns: int = 4) -> str:
    """Extracts the first few turns for quick reference while labeling."""
    if pd.isna(conversation):
        return ""
    turns = conversation.split("|EOM|")
    cleaned = []
    for turn in turns[:max_turns]:
        turn = turn.strip()
        if not turn:
            continue
        # strip the (dialog_act) tag for readability, keep persona + text
        match = re.match(r"(Teacher|Student):\s*(\([^)]*\))?\s*(.*)", turn)
        if match:
            persona, _, text = match.groups()
            cleaned.append(f"{persona}: {text.strip()}")
        else:
            cleaned.append(turn)
    return " || ".join(cleaned)

def has_usable_confusion(row) -> bool:
    confusion = row.get("teacher_described_confusion", "")
    return isinstance(confusion, str) and len(confusion.strip()) > 0

def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[df["question"].apply(matches_topic)]

    if df.empty:
        print("No matches — check TOPIC_KEYWORDS against real data.")
        return

    df["label_type"] = df.apply(
        lambda row: "diagnosable" if has_usable_confusion(row) else "off_topic",
        axis=1,
    )
    df["conversation_excerpt"] = df["conversation"].apply(parse_first_turns)

    out = pd.DataFrame({
        "problem_id": df["qid"],
        "question": df["question"],
        "ground_truth": df["ground_truth"],
        "student_said": df["student_incorrect_solution"],
        "student_profile": df["student_profile"],
        "conversation_excerpt": df["conversation_excerpt"],
        "self_typical_confusion": df.get("self-typical-confusion", ""),
        "label_type": df["label_type"],
        "inferred_gap": df["teacher_described_confusion"],  # pre-filled
        "dimension": "",                                     # fill by hand
    })

    sample = out.sample(n=min(SAMPLE_SIZE, len(out)), random_state=RANDOM_SEED)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(sample)} rows (from {len(out)} topic matches) to {OUTPUT_PATH}")
    print(sample["label_type"].value_counts())

if __name__ == "__main__":
    main()