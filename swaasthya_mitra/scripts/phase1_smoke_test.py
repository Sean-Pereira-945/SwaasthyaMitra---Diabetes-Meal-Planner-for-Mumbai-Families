from pathlib import Path
import importlib
import os
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    ROOT / "data" / "pdfs",
    ROOT / "data" / "csvs",
    ROOT / "data" / "chroma_db",
    ROOT / "outputs",
    ROOT / "evaluation",
]

REQUIRED_FILES = [
    ROOT / "app.py",
    ROOT / "requirements.txt",
    ROOT / "README.md",
    ROOT / ".env",
]

REQUIRED_IMPORTS = [
    "streamlit",
    "pandas",
    "langchain",
    "langchain_community",
    "langchain_groq",
    "chromadb",
    "sentence_transformers",
    "pypdf",
    "reportlab",
    "gtts",
    "dotenv",
]


def check_paths() -> list[str]:
    errors: list[str] = []

    for d in REQUIRED_DIRS:
        if not d.exists() or not d.is_dir():
            errors.append(f"Missing directory: {d}")

    for f in REQUIRED_FILES:
        if not f.exists() or not f.is_file():
            errors.append(f"Missing file: {f}")

    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"Import failed for {name}: {exc}")
    return errors


def check_env() -> list[str]:
    errors: list[str] = []
    env_file = ROOT / ".env"

    if not env_file.exists():
        errors.append(".env file missing")
        return errors

    content = env_file.read_text(encoding="utf-8", errors="ignore")
    if "GROQ_API_KEY=" not in content:
        errors.append("GROQ_API_KEY entry missing in .env")

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        print("Warning: GROQ_API_KEY is not loaded in shell environment. Generation requires a valid key in .env.")

    return errors


def main() -> int:
    all_errors = []
    all_errors.extend(check_paths())
    all_errors.extend(check_imports())
    all_errors.extend(check_env())

    if all_errors:
        print("Phase 1 smoke test found issues:")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print("Phase 1 smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
