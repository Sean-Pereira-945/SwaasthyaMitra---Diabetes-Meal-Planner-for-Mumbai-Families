import logging
import os
import re
import time
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "data" / "chroma_db"


def build_chain() -> RetrievalQA:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        client_settings=Settings(anonymized_telemetry=False),
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are SwaasthyaMitra, a diabetes nutrition assistant for Indian families.\n"
            "Use the context to answer accurately and safely.\n"
            "If the context is partial, make cautious assumptions and state them.\n"
            "Avoid recommending direct sugar terms for diabetic users.\n\n"
            "If asked about sugar or jaggery, explicitly state they should be avoided or strictly limited.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1200,
        api_key=api_key,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": prompt},
    )


def run_checks() -> list[str]:
    chain = build_chain()

    test_cases = [
        {
            "id": "low_gi_threshold",
            "question": "What is generally considered low GI for diabetic meal planning?",
            "required": [r"\b55\b|<\s*55|less than\s*55"],
            "forbidden": [r"i\s+don't\s+know", r"cannot\s+answer"],
        },
        {
            "id": "sugar_safety",
            "question": "Should sugar or jaggery be included in a diabetic meal plan?",
            "required": [r"avoid|limit|not recommended|should not"],
            "forbidden": [r"recommend\s+sugar", r"add\s+jaggery\s+daily"],
        },
        {
            "id": "meal_structure",
            "question": "What meal slots should a 7-day diabetic meal plan include each day?",
            "required": [r"breakfast", r"lunch", r"dinner", r"snack|mid-morning"],
            "forbidden": [r"i\s+don't\s+know", r"cannot\s+answer"],
        },
    ]

    failures: list[str] = []

    for case in test_cases:
        start = time.perf_counter()
        answer = chain.invoke({"query": case["question"]}).get("result", "")
        elapsed = round(time.perf_counter() - start, 2)

        print(f"case={case['id']} latency_s={elapsed} chars={len(answer)}")

        if not answer.strip():
            failures.append(f"{case['id']}: empty answer")
            continue

        answer_lc = answer.lower()

        for pat in case["required"]:
            if not re.search(pat, answer_lc, flags=re.IGNORECASE):
                failures.append(f"{case['id']}: missing required pattern {pat}")

        for pat in case["forbidden"]:
            if re.search(pat, answer_lc, flags=re.IGNORECASE):
                failures.append(f"{case['id']}: matched forbidden pattern {pat}")

    return failures


def main() -> int:
    failures = run_checks()
    if failures:
        print("rag_chatbot_check: FAILED")
        for failure in failures:
            print("-", failure)
        return 1

    print("rag_chatbot_check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
