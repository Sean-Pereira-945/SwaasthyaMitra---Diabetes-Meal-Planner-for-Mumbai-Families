import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"


def load_validator_functions():
    source = APP_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_FILE))

    fn_names = {"parse_numeric_values", "validate_plan"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in fn_names]

    if len(selected) != 2:
        raise RuntimeError("Could not locate parse_numeric_values and validate_plan in app.py")

    module = ast.Module(body=[node for node in selected], type_ignores=[])
    namespace = {"re": __import__("re"), "Any": object}
    exec(compile(module, str(APP_FILE), "exec"), namespace)
    return namespace["validate_plan"]


def run_edge_tests(validate_plan):
    cases = [
        {
            "name": "high_gi_and_sugar",
            "text": "Day 1 Breakfast 78 GI 320 calories includes sugar tea and jaggery sweet. Lunch 74 GI 500 calories.",
            "age": 45,
            "bs": 170,
            "expect_flags": ["contains_high_gi_items", "contains_direct_sugar_terms"],
            "expect_safe": False,
        },
        {
            "name": "missing_calories_and_coverage",
            "text": "Day 1 Breakfast 45 GI. Day 2 Breakfast 50 GI. Day 3 Breakfast 49 GI.",
            "age": 52,
            "bs": 190,
            "expect_flags": ["missing_calorie_values", "incomplete_meal_coverage"],
            "expect_safe": False,
        },
        {
            "name": "safe_structured_plan",
            "text": (
                "Day 1 Breakfast GI 42 250 kcal Mid-morning GI 35 120 kcal Lunch GI 48 420 kcal Evening snack GI 40 140 kcal Dinner GI 50 380 kcal. "
                "Day 2 Breakfast 45 GI 260 kcal Mid-morning 38 GI 110 kcal Lunch 50 GI 430 kcal Evening snack 39 GI 130 kcal Dinner 52 GI 390 kcal. "
                "Day 3 Breakfast 43 GI 255 kcal Mid-morning 36 GI 115 kcal Lunch 49 GI 425 kcal Evening snack 41 GI 135 kcal Dinner 51 GI 385 kcal. "
                "Day 4 Breakfast 44 GI 250 kcal Mid-morning 37 GI 110 kcal Lunch 50 GI 420 kcal Evening snack 40 GI 130 kcal Dinner 52 GI 390 kcal. "
                "Day 5 Breakfast 42 GI 245 kcal Mid-morning 35 GI 115 kcal Lunch 48 GI 430 kcal Evening snack 39 GI 140 kcal Dinner 51 GI 395 kcal. "
                "Day 6 Breakfast 43 GI 250 kcal Mid-morning 36 GI 110 kcal Lunch 49 GI 425 kcal Evening snack 40 GI 135 kcal Dinner 52 GI 385 kcal. "
                "Day 7 Breakfast 44 GI 255 kcal Mid-morning 37 GI 115 kcal Lunch 50 GI 430 kcal Evening snack 41 GI 130 kcal Dinner 53 GI 390 kcal."
            ),
            "age": 38,
            "bs": 145,
            "expect_flags": [],
            "expect_safe": True,
        },
    ]

    failures = []

    for case in cases:
        result = validate_plan(case["text"], case["age"], case["bs"])
        violations = set(v.strip() for v in str(result.get("violations", "")).split(",") if v.strip() and v.strip() != "none")

        for expected_flag in case["expect_flags"]:
            if expected_flag not in violations:
                failures.append(f"{case['name']}: missing expected violation {expected_flag}")

        safe = bool(result.get("medically_safe", False))
        if safe != case["expect_safe"]:
            failures.append(f"{case['name']}: expected medically_safe={case['expect_safe']} got {safe}")

        print(
            f"case={case['name']} safe={safe} hard_fail_count={result.get('hard_fail_count')} violations={result.get('violations')}"
        )

    return failures


def main() -> int:
    validate_plan = load_validator_functions()
    failures = run_edge_tests(validate_plan)

    if failures:
        print("phase3_validator_check: FAILED")
        for failure in failures:
            print("-", failure)
        return 1

    print("phase3_validator_check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
