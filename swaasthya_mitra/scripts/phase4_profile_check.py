import ast
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"


def load_history_functions():
    source = APP_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    needed = [
        "normalize_profile_key",
        "list_profiles",
        "append_history",
        "load_full_history",
    ]

    fn_src = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in needed:
            segment = ast.get_source_segment(source, node)
            if segment:
                fn_src[node.name] = segment

    missing = [name for name in needed if name not in fn_src]
    if missing:
        raise RuntimeError(f"Missing functions in app.py: {missing}")

    namespace = {
        "pd": pd,
        "datetime": datetime,
        "timedelta": timedelta,
        "re": __import__("re"),
    }

    exec("\n\n".join(fn_src[name] for name in needed), namespace)
    return namespace


def run_checks() -> list[str]:
    failures: list[str] = []
    ns = load_history_functions()

    normalize_profile_key = ns["normalize_profile_key"]
    list_profiles = ns["list_profiles"]
    append_history = ns["append_history"]
    load_full_history = ns["load_full_history"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ns["OUTPUTS_DIR"] = tmp_path
        ns["HISTORY_CSV"] = tmp_path / "plan_history.csv"

        empty_profiles = list_profiles(pd.DataFrame())
        if empty_profiles != ["Default Family"]:
            failures.append(f"Expected default profile for empty history, got: {empty_profiles}")

        now = datetime.now()
        synthetic_history = pd.DataFrame(
            [
                {
                    "created_at": now.isoformat(),
                    "profile_key": normalize_profile_key("Patel Family"),
                    "profile_name": "Patel Family",
                    "mode": "RAG",
                    "age": 45,
                    "blood_sugar": 150,
                    "festival": "None",
                    "validator_score": 0.82,
                    "avg_gi": 49.0,
                    "medically_safe": True,
                },
                {
                    "created_at": now.isoformat(),
                    "profile_key": normalize_profile_key("Khan Family"),
                    "profile_name": "Khan Family",
                    "mode": "Baseline",
                    "age": 52,
                    "blood_sugar": 165,
                    "festival": "Eid",
                    "validator_score": 0.76,
                    "avg_gi": 54.0,
                    "medically_safe": True,
                },
            ]
        )

        # Match app-side profile filtering logic.
        a_count = len(synthetic_history[synthetic_history["profile_name"].astype(str) == "Patel Family"])
        b_count = len(synthetic_history[synthetic_history["profile_name"].astype(str) == "Khan Family"])
        if a_count != 1 or b_count != 1:
            failures.append(f"Profile isolation failed: Patel={a_count}, Khan={b_count}")

        profiles = sorted(list_profiles(synthetic_history))
        if profiles != ["Khan Family", "Patel Family"]:
            failures.append(f"Unexpected profile list values: {profiles}")

        old_df = pd.DataFrame(
            [
                {
                    "created_at": (now - timedelta(days=40)).isoformat(),
                    "profile_key": "old_profile",
                    "profile_name": "Old Family",
                    "mode": "RAG",
                    "age": 60,
                    "blood_sugar": 170,
                    "festival": "None",
                    "validator_score": 0.60,
                    "avg_gi": 58.0,
                    "medically_safe": False,
                },
                {
                    "created_at": (now - timedelta(days=10)).isoformat(),
                    "profile_key": "recent_profile",
                    "profile_name": "Recent Family",
                    "mode": "RAG",
                    "age": 41,
                    "blood_sugar": 145,
                    "festival": "None",
                    "validator_score": 0.85,
                    "avg_gi": 48.0,
                    "medically_safe": True,
                },
            ]
        )
        old_df.to_csv(ns["HISTORY_CSV"], index=False)

        append_history(
            {
                "created_at": now.isoformat(),
                "profile_key": "new_profile",
                "profile_name": "New Family",
                "mode": "RAG",
                "age": 35,
                "blood_sugar": 140,
                "festival": "None",
                "validator_score": 0.88,
                "avg_gi": 47.0,
                "medically_safe": True,
            }
        )

        pruned = load_full_history()
        if (pruned["profile_name"].astype(str) == "Old Family").any():
            failures.append("30-day pruning failed: old history row still present")

        if len(pruned) != 2:
            failures.append(f"Expected 2 rows after pruning, found {len(pruned)}")

    return failures


def main() -> int:
    failures = run_checks()
    if failures:
        print("phase4_profile_check: FAILED")
        for failure in failures:
            print("-", failure)
        return 1

    print("phase4_profile_check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
