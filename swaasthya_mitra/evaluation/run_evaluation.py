import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader, PyPDFDirectoryLoader
from langchain_groq import ChatGroq

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"
CSV_DIR = ROOT / "data" / "csvs"
CHROMA_DIR = ROOT / "data" / "chroma_db"
CASE_CSV = ROOT / "evaluation" / "test_cases.csv"
OUT_CSV = ROOT / "evaluation" / "test_results.csv"

CSV_FILES = [
    CSV_DIR / "indian_nutrition_2025.csv",
    CSV_DIR / "indian_recipes_2026.csv",
]


def parse_numeric_values(text: str, token: str) -> list[float]:
    pattern = re.compile(rf"(\d+(?:\.\d+)?)\s*{token}", re.IGNORECASE)
    return [float(match.group(1)) for match in pattern.finditer(text)]


def heuristic_scores(plan_text: str) -> dict[str, float]:
    gi_values = parse_numeric_values(plan_text, r"(?:gi|glycemic index)")
    calorie_values = parse_numeric_values(plan_text, r"(?:kcal|calories?)")

    meal_words = ["breakfast", "lunch", "dinner", "snack", "mid-morning"]
    meal_coverage = sum(1 for word in meal_words if word in plan_text.lower()) / len(meal_words)

    low_gi_ratio = sum(1 for val in gi_values if val < 55) / len(gi_values) if gi_values else 0.0
    calorie_presence = 1.0 if calorie_values else 0.0

    accuracy = (0.6 * low_gi_ratio) + (0.4 * calorie_presence)
    adherence = (0.5 * meal_coverage) + (0.5 * ("festival" in plan_text.lower() or "local" in plan_text.lower()))
    safety = (0.7 * low_gi_ratio) + (0.3 * meal_coverage)

    hard_fail_count = 0
    if not gi_values:
        hard_fail_count += 1
    if gi_values and max(gi_values) > 70:
        hard_fail_count += 1
    if meal_coverage < 0.8:
        hard_fail_count += 1
    if any(term in plan_text.lower() for term in ["sugar", "jaggery", "sweet syrup"]):
        hard_fail_count += 1

    return {
        "gold_accuracy": round(float(accuracy), 3),
        "adherence_score": round(float(adherence), 3),
        "safety_score": round(float(safety), 3),
        "hard_fail_count": float(hard_fail_count),
        "medically_safe": 1.0 if hard_fail_count == 0 else 0.0,
    }


def build_query(row: pd.Series) -> str:
    return f"""You are a domain expert diabetes nutritionist for urban Indian families.
Create a detailed 7-day low-GI (<55) diabetic meal plan for age {int(row['age'])}, blood sugar {int(row['bs'])} mg/dL, budget {row['budget']}.
Festival: {row['festival']}. Preferences: {row['preferences']}.
Use only Mumbai-local, affordable, seasonal ingredients. For each day provide:
- Breakfast, Mid-morning, Lunch, Evening snack, Dinner
- Approximate GI value and calories per meal
- Grocery list for the week with estimated Mumbai prices in INR
- Marathi/Hindi-friendly explanations where possible
Emphasize cultural relevance and festival adjustments.
Add a short adherence tip for each day."""


def load_rag_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_settings = Settings(anonymized_telemetry=False)

    existing_db_files = [p for p in CHROMA_DIR.glob("**/*") if p.is_file() and p.name != ".gitkeep"]
    if existing_db_files:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
            client_settings=chroma_settings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 6})

    docs = []
    docs.extend(PyPDFDirectoryLoader(str(PDF_DIR)).load())
    for csv_file in CSV_FILES:
        docs.extend(CSVLoader(str(csv_file)).load())

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        client_settings=chroma_settings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")

    if not CASE_CSV.exists():
        raise FileNotFoundError(f"Missing case CSV: {CASE_CSV}")

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1200,
        api_key=api_key,
    )
    retriever = load_rag_retriever()

    cases = pd.read_csv(CASE_CSV)
    results = []

    for _, row in cases.iterrows():
        mode = str(row.get("case_type", "baseline")).strip().lower()
        query = build_query(row)

        start = time.perf_counter()
        if mode == "rag":
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            answer = chain.invoke({"query": query}).get("result", "")
        else:
            answer = llm.invoke("Answer without external retrieval.\n\n" + query).content
        latency = time.perf_counter() - start

        scores = heuristic_scores(answer)
        results.append(
            {
                "case_id": row.get("case_id"),
                "case_type": mode,
                "age": row.get("age"),
                "bs": row.get("bs"),
                "budget": row.get("budget"),
                "festival": row.get("festival"),
                "preferences": row.get("preferences"),
                "gold_accuracy": scores["gold_accuracy"],
                "adherence_score": scores["adherence_score"],
                "safety_score": scores["safety_score"],
                "hard_fail_count": int(scores["hard_fail_count"]),
                "medically_safe": int(scores["medically_safe"]),
                "latency_sec": round(float(latency), 3),
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved evaluation outputs to: {OUT_CSV}")


if __name__ == "__main__":
    main()
