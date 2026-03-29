import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader, PyPDFDirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

load_dotenv()
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdfs"
CSV_DIR = ROOT / "data" / "csvs"
CHROMA_DIR = ROOT / "data" / "chroma_db"
OUTPUTS_DIR = ROOT / "outputs"
HISTORY_CSV = OUTPUTS_DIR / "plan_history.csv"
PROFILES_CSV = OUTPUTS_DIR / "profiles.csv"
PRICE_CSV = CSV_DIR / "mumbai_prices.csv"

CSV_FILES = [
    CSV_DIR / "indian_nutrition_2025.csv",
    CSV_DIR / "indian_recipes_2026.csv",
]

st.set_page_config(page_title="SwaasthyaMitra", page_icon="SM", layout="wide")
st.title("SwaasthyaMitra - Diabetes Meal Planner for Mumbai Families")
st.caption("RAG-backed low-GI planning with local context, adherence-oriented advice, and evaluation support.")


def get_missing_data_files() -> list[str]:
    missing = []

    if not PDF_DIR.exists() or not any(PDF_DIR.glob("*.pdf")):
        missing.append("data/pdfs/*.pdf")

    for csv_file in CSV_FILES:
        if not csv_file.exists():
            missing.append(str(csv_file.relative_to(ROOT)).replace("\\", "/"))

    return missing


def load_price_map() -> dict[str, float]:
    if not PRICE_CSV.exists():
        return {}

    try:
        df = pd.read_csv(PRICE_CSV)
    except Exception:
        return {}

    if "item" not in df.columns or "price_per_kg" not in df.columns:
        return {}

    df = df.dropna(subset=["item", "price_per_kg"])
    return {str(row["item"]).strip().lower(): float(row["price_per_kg"]) for _, row in df.iterrows()}


@st.cache_resource(show_spinner="Building knowledge base (first run takes 30-90s)...")
def load_rag():
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
    skipped_sources: list[str] = []

    # Try fast directory parsing first, then fall back to per-file parsing if a PDF is malformed.
    try:
        pdf_loader = PyPDFDirectoryLoader(str(PDF_DIR))
        docs.extend(pdf_loader.load())
    except Exception:
        for pdf_file in PDF_DIR.glob("*.pdf"):
            try:
                docs.extend(PyPDFLoader(str(pdf_file)).load())
            except Exception as exc:
                skipped_sources.append(f"{pdf_file.name}: {exc}")

    for csv_file in CSV_FILES:
        try:
            docs.extend(CSVLoader(str(csv_file)).load())
        except Exception as exc:
            skipped_sources.append(f"{csv_file.name}: {exc}")

    if not docs:
        raise RuntimeError("No documents could be loaded from data/pdfs or data/csvs.")

    if skipped_sources:
        print("Skipped unreadable sources:")
        for item in skipped_sources[:10]:
            print(f"- {item}")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        client_settings=chroma_settings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})


def build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_groq_key":
        raise RuntimeError("Set GROQ_API_KEY in .env before generating a meal plan.")

    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1200,
        api_key=api_key,
    )


def parse_numeric_values(text: str, token: str) -> list[float]:
    # Accept both formats: "40 GI" and "GI: 40" (same for calories).
    pattern_after = re.compile(rf"(\d+(?:\.\d+)?)\s*{token}", re.IGNORECASE)
    pattern_before = re.compile(rf"{token}\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

    values = [float(match.group(1)) for match in pattern_after.finditer(text)]
    values.extend(float(match.group(1)) for match in pattern_before.finditer(text))
    return values


def normalize_profile_key(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return cleaned or "default_profile"


def list_profiles(history: pd.DataFrame) -> list[str]:
    if history.empty or "profile_name" not in history.columns:
        return ["Default Family"]
    values = [str(v).strip() for v in history["profile_name"].dropna().tolist() if str(v).strip()]
    if not values:
        return ["Default Family"]
    return sorted(set(values))


def validate_plan(plan_text: str, age: int, blood_sugar: int) -> dict[str, Any]:
    gi_values = parse_numeric_values(plan_text, r"(?:gi|glycemic index)")
    calorie_values = parse_numeric_values(plan_text, r"(?:kcal|calories?)")

    meal_words = ["breakfast", "lunch", "dinner", "snack", "mid-morning"]
    meal_coverage = sum(1 for word in meal_words if word in plan_text.lower()) / len(meal_words)

    violations: list[str] = []

    if gi_values:
        low_gi_ratio = sum(1 for val in gi_values if val < 55) / len(gi_values)
        avg_gi = sum(gi_values) / len(gi_values)
    else:
        low_gi_ratio = 0.0
        avg_gi = 0.0
        violations.append("missing_gi_values")

    if avg_gi > 55:
        violations.append("avg_gi_above_target")
    if gi_values and max(gi_values) > 70:
        violations.append("contains_high_gi_items")
    if meal_coverage < 0.8:
        violations.append("incomplete_meal_coverage")

    if calorie_values:
        daily_avg = sum(calorie_values) / 7 if len(calorie_values) >= 7 else sum(calorie_values)
    else:
        daily_avg = 0.0
        violations.append("missing_calorie_values")

    cal_min = 1100 if blood_sugar >= 180 else 1000
    cal_max = 2000 if blood_sugar >= 180 else 2200
    if daily_avg and (daily_avg < cal_min or daily_avg > cal_max):
        violations.append("calorie_range_outside_target")

    sugar_terms = ["sugar", "jaggery", "mithai", "sweet syrup"]
    if any(term in plan_text.lower() for term in sugar_terms):
        violations.append("contains_direct_sugar_terms")

    age_risk = "senior" if age >= 60 else "adult"

    # Overall score combines low GI consistency, meal completeness, and calorie presence.
    calorie_presence = 1.0 if calorie_values else 0.0
    penalty = min(len(violations) * 0.08, 0.4)
    overall = ((0.5 * low_gi_ratio) + (0.3 * meal_coverage) + (0.2 * calorie_presence)) - penalty
    overall = max(0.0, min(1.0, overall))

    risk_flag = "ok"
    if "contains_high_gi_items" in violations or "contains_direct_sugar_terms" in violations:
        risk_flag = "high"
    elif gi_values and avg_gi > 58:
        risk_flag = "moderate"
    elif gi_values and avg_gi > 60:
        risk_flag = "high_gi_risk"
    elif calorie_values and daily_avg < 900:
        risk_flag = "low_calorie_risk"

    hard_fail_count = sum(
        1
        for rule in ["missing_gi_values", "contains_high_gi_items", "contains_direct_sugar_terms", "incomplete_meal_coverage"]
        if rule in violations
    )
    medically_safe = hard_fail_count == 0

    return {
        "overall_score": round(overall, 3),
        "avg_gi": round(avg_gi, 2),
        "low_gi_ratio": round(low_gi_ratio, 3),
        "meal_coverage": round(meal_coverage, 3),
        "daily_calorie_estimate": round(daily_avg, 2),
        "target_calorie_min": cal_min,
        "target_calorie_max": cal_max,
        "violations": ", ".join(violations) if violations else "none",
        "hard_fail_count": hard_fail_count,
        "medically_safe": medically_safe,
        "age_risk_group": age_risk,
        "risk_flag": risk_flag,
    }


def append_history(row: dict[str, object]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    if HISTORY_CSV.exists():
        old = pd.read_csv(HISTORY_CSV)
        combined = pd.concat([old, row_df], ignore_index=True)
    else:
        combined = row_df

    cutoff = datetime.now() - timedelta(days=30)
    combined["created_at"] = pd.to_datetime(combined["created_at"], errors="coerce")
    combined = combined[combined["created_at"] >= cutoff]
    combined.to_csv(HISTORY_CSV, index=False)


def load_full_history() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(HISTORY_CSV)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    return df


def load_recent_history() -> pd.DataFrame:
    df = load_full_history()
    if df.empty:
        return df
    return df.sort_values("created_at", ascending=False).head(20)


def load_profile_registry() -> pd.DataFrame:
    if not PROFILES_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(PROFILES_CSV)
    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return df


def upsert_profile(details: dict[str, object]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([details])

    if PROFILES_CSV.exists():
        existing = pd.read_csv(PROFILES_CSV)
        if "profile_key" in existing.columns:
            existing = existing[existing["profile_key"].astype(str) != str(details["profile_key"])].copy()
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(PROFILES_CSV, index=False)


def build_query(age: float | int, blood_sugar: float | int, budget: str | None, festival: str, preferences: str) -> str:
    safe_budget = budget if budget else "500-800"
    return f"""You are a domain expert diabetes nutritionist for urban Indian families.
Create a detailed 7-day low-GI (<55) diabetic meal plan for age {int(age)}, blood sugar {int(blood_sugar)} mg/dL, budget {safe_budget}.
Festival: {festival}. Preferences: {preferences}.
Use only Mumbai-local, affordable, seasonal ingredients. For each day provide:
- Breakfast, Mid-morning, Lunch, Evening snack, Dinner
- Approximate GI value and calories per meal
- Grocery list for the week with estimated Mumbai prices in INR
- Marathi/Hindi-friendly explanations where possible
Emphasize cultural relevance and festival adjustments.
Add a short adherence tip for each day."""


def has_complete_7_day_structure(text: str) -> bool:
    day_headers = set(
        int(match.group(1))
        for match in re.finditer(r"\bday\s*([1-7])\b", text, flags=re.IGNORECASE)
    )
    return day_headers == {1, 2, 3, 4, 5, 6, 7}


def enforce_7_day_structure(llm: ChatGroq, text: str) -> str:
    if not text.strip() or has_complete_7_day_structure(text):
        return text

    repair_prompt = (
        "Rewrite the plan into an explicit 7-day format with headings exactly as Day 1 through Day 7. "
        "For each day include Breakfast, Mid-morning, Lunch, Evening snack, Dinner, approximate GI and calories, "
        "and one adherence tip. Preserve low-GI diabetic safety intent and avoid adding direct sugar terms.\n\n"
        f"Original plan:\n{text}"
    )
    repaired = llm.invoke(repair_prompt).content
    return repaired if repaired.strip() else text


def strip_existing_grocery_section(text: str) -> str:
    pattern = re.compile(r"\n\s*grocery\s+list.*$", flags=re.IGNORECASE | re.DOTALL)
    return re.sub(pattern, "", text).strip()


def parse_budget_cap(budget: str | None) -> int:
    if not budget:
        return 800
    cleaned = budget.strip().lower()
    if cleaned == "<500":
        return 500
    if cleaned == "500-800":
        return 800
    if cleaned == "800-1200":
        return 1200
    if cleaned == ">1200":
        return 1500
    return 800


def build_price_hint(price_map: dict[str, float]) -> str:
    if not price_map:
        return "No custom local price file provided. Use typical Mumbai retail estimates."

    pairs = sorted(price_map.items())[:20]
    lines = [f"- {name}: {price:.2f} INR/kg" for name, price in pairs]
    return "Known local prices (use when relevant):\n" + "\n".join(lines)


def generate_grocery_section(
    llm: ChatGroq,
    plan_text: str,
    budget: str | None,
    festival: str,
    preferences: str,
    price_map: dict[str, float],
) -> str:
    budget_cap = parse_budget_cap(budget)
    price_hint = build_price_hint(price_map)
    prompt = (
        "Create ONLY a grocery list section for the weekly meal plan below.\n"
        "Requirements:\n"
        "1) Output must start with exactly: Grocery List (Estimated Mumbai Prices in INR):\n"
        "2) Then output a markdown table with columns: Item | Weekly Qty | Estimated Price (INR) | Notes\n"
        "3) Include 12-18 realistic items maximum.\n"
        "4) Keep total weekly estimate <= budget cap when possible.\n"
        "5) Respect dietary constraints from the plan and preferences.\n"
        "6) Do not include GI or calorie values in this section.\n"
        "7) End with: Estimated Weekly Total: ₹<amount>\n\n"
        f"Budget bracket: {budget or '500-800'} (cap {budget_cap} INR)\n"
        f"Festival: {festival}\n"
        f"Preferences: {preferences}\n"
        f"{price_hint}\n\n"
        f"Plan:\n{plan_text}"
    )

    try:
        response = llm.invoke(prompt).content.strip()
    except Exception:
        response = ""

    if not response:
        return (
            "Grocery List (Estimated Mumbai Prices in INR):\n"
            "| Item | Weekly Qty | Estimated Price (INR) | Notes |\n"
            "|---|---|---:|---|\n"
            "| Oats | 500 g | 80 | Breakfast base |\n"
            "| Brown rice | 1 kg | 90 | Low-GI staple |\n"
            "| Whole wheat atta | 2 kg | 130 | Roti base |\n"
            "| Mixed lentils | 1.5 kg | 220 | Dal and protein |\n"
            "| Seasonal vegetables | 4 kg | 320 | Mixed sabzi and salads |\n"
            "| Curd | 1.5 kg | 120 | Raita and snacks |\n"
            "| Tofu/Paneer (low-fat) | 500 g | 180 | Protein option |\n"
            "| Makhana/roasted chana | 500 g | 140 | Snack option |\n"
            "Estimated Weekly Total: ₹1280"
        )

    if "grocery list" not in response.lower():
        response = "Grocery List (Estimated Mumbai Prices in INR):\n" + response

    return response


def ensure_complete_grocery_list(
    llm: ChatGroq,
    plan_text: str,
    budget: str | None,
    festival: str,
    preferences: str,
    price_map: dict[str, float],
) -> str:
    base_text = strip_existing_grocery_section(plan_text)
    grocery_section = generate_grocery_section(llm, base_text, budget, festival, preferences, price_map)
    return f"{base_text}\n\n{grocery_section}".strip()


def run_baseline(llm: ChatGroq, query: str) -> str:
    baseline_prompt = (
        "Answer directly without any external retrieval. "
        "Generate a complete 7-day plan with headings Day 1 to Day 7. "
        "Include reasonable assumptions and confidence caveats for uncertain GI values.\n\n"
        f"{query}"
    )
    raw = llm.invoke(baseline_prompt).content
    return enforce_7_day_structure(llm, raw)


def run_rag(llm: ChatGroq, retriever, query: str) -> str:
    rag_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are SwaasthyaMitra, a diabetes nutrition assistant for Indian families.\n"
            "Use the provided context to answer the user request accurately and safely.\n"
            "If context is partial, use conservative assumptions and state them clearly.\n"
            "Never recommend direct sugar terms (sugar, jaggery, sweet syrup) for diabetic plans.\n"
            "If asked about sugar/jaggery, explicitly state they should be avoided or strictly limited.\n"
            "Return practical, concise, medically cautious guidance.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt},
    )
    rag_query = (
        query
        + "\n\nUse retrieved evidence when available. If evidence is partial, still provide a complete 7-day plan "
        "with headings Day 1 to Day 7 and clearly stated assumptions. Do not refuse the request."
    )
    result = qa_chain.invoke({"query": rag_query})
    raw = result.get("result", "")
    return enforce_7_day_structure(llm, raw)


def extract_overall_score(validation: dict[str, float | str]) -> float:
    score = validation.get("overall_score", 0.0)
    return float(score) if isinstance(score, (int, float, str)) else 0.0


def save_pdf(plan_text: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUTS_DIR / f"swaasthya_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(50, 760, "SwaasthyaMitra - Personalized Diabetes Meal Plan")
    c.drawString(50, 742, f"Date: {datetime.now().strftime('%d %B %Y')}")

    y = 720
    max_lines = 55
    for idx, line in enumerate(plan_text.splitlines()):
        if idx >= max_lines or y < 60:
            break
        c.drawString(50, y, line[:95])
        y -= 12

    c.save()
    return pdf_path


def split_text_for_tts(text: str, max_chars: int = 1400) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
        else:
            for i in range(0, len(sentence), max_chars):
                part = sentence[i : i + max_chars]
                if len(part) == max_chars:
                    chunks.append(part)
                else:
                    current = part
    if current:
        chunks.append(current)
    return chunks[:8]


def save_voice_clips(plan_text: str, label: str = "plan") -> list[Path]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    clips: list[Path] = []
    for idx, chunk in enumerate(split_text_for_tts(plan_text), start=1):
        clip_path = OUTPUTS_DIR / f"voice_{label}_{timestamp}_part{idx}.mp3"
        tts = gTTS(chunk, lang="hi")
        tts.save(str(clip_path))
        clips.append(clip_path)
    return clips


missing_files = get_missing_data_files()
if missing_files:
    st.warning("Missing data files. Add these before running RAG:")
    for item in missing_files:
        st.write(f"- {item}")
    st.stop()

retriever = None
llm = None

try:
    retriever = load_rag()
    llm = build_llm()
except Exception as exc:
    st.error(f"Startup error: {exc}")
    st.stop()

if retriever is None or llm is None:
    st.error("Startup failed to initialize retriever or LLM.")
    st.stop()

st.sidebar.header("Generation Settings")
mode = st.sidebar.radio("Mode", ["RAG", "Baseline", "Compare Both"], index=0)
k_value = st.sidebar.slider("Retriever k", min_value=3, max_value=8, value=6)

if k_value != 6:
    retriever.search_kwargs["k"] = k_value

price_map = load_price_map()
if price_map:
    st.sidebar.success("Mumbai price file detected.")
else:
    st.sidebar.info("Optional: add data/csvs/mumbai_prices.csv (item, price_per_kg).")

full_history_df = load_full_history()
profile_registry_df = load_profile_registry()

if "custom_profiles" not in st.session_state:
    st.session_state["custom_profiles"] = []

history_profiles = list_profiles(full_history_df)
saved_profiles: list[str] = []
if not profile_registry_df.empty and "profile_name" in profile_registry_df.columns:
    saved_profiles = [
        str(v).strip() for v in profile_registry_df["profile_name"].dropna().tolist() if str(v).strip()
    ]

new_profile_name = st.sidebar.text_input("Create/Use New Profile", "")
typed_profile = new_profile_name.strip()

if typed_profile and typed_profile not in history_profiles and typed_profile not in st.session_state["custom_profiles"]:
    st.session_state["custom_profiles"].append(typed_profile)

available_profiles = sorted(set(history_profiles + saved_profiles + st.session_state["custom_profiles"]))
default_profile = typed_profile if typed_profile else available_profiles[0]
default_index = available_profiles.index(default_profile) if default_profile in available_profiles else 0
selected_profile = st.sidebar.selectbox("Family Profile", available_profiles, index=default_index)
active_profile = typed_profile if typed_profile else selected_profile
active_profile_key = normalize_profile_key(active_profile)

saved_profile_row: dict[str, Any] = {}
if not profile_registry_df.empty and "profile_key" in profile_registry_df.columns:
    match = profile_registry_df[profile_registry_df["profile_key"].astype(str) == active_profile_key].copy()
    if not match.empty:
        if "updated_at" in match.columns:
            match = match.sort_values("updated_at", ascending=False)
        saved_profile_row = match.iloc[0].to_dict()

if not full_history_df.empty and "profile_name" in full_history_df.columns:
    profile_history = full_history_df[
        full_history_df["profile_name"].astype(str).str.strip() == active_profile.strip()
    ].copy()
else:
    profile_history = pd.DataFrame()

left, right = st.columns([2, 1])

with left:
    # Reset form values whenever the active profile changes.
    if st.session_state.get("active_profile_form_key") != active_profile_key:
        st.session_state["active_profile_form_key"] = active_profile_key
        st.session_state["age_input"] = int(saved_profile_row.get("age", 45) or 45)
        st.session_state["blood_sugar_input"] = int(saved_profile_row.get("blood_sugar", 140) or 140)
        st.session_state["budget_input"] = str(saved_profile_row.get("budget", "500-800") or "500-800")
        st.session_state["festival_input"] = str(saved_profile_row.get("festival", "None") or "None")
        st.session_state["preferences_input"] = str(
            saved_profile_row.get(
                "preferences",
                "Maharashtrian style, Jain preferred, affordable Dadar market ingredients",
            )
            or "Maharashtrian style, Jain preferred, affordable Dadar market ingredients"
        )

    budget_options = ["<500", "500-800", "800-1200", ">1200"]
    if st.session_state["budget_input"] not in budget_options:
        st.session_state["budget_input"] = "500-800"

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, key="age_input")
        blood_sugar = st.number_input(
            "Current fasting blood sugar (mg/dL)",
            min_value=70,
            max_value=300,
            key="blood_sugar_input",
        )
    with col2:
        budget = st.selectbox("Weekly food budget (INR)", budget_options, key="budget_input")
        festival = st.text_input("Upcoming festival (or None)", key="festival_input")

    preferences = st.text_area(
        "Preferences, allergies, family style",
        key="preferences_input",
    )

    st.info(
        "This tool is educational support, not medical advice. "
        "Users should validate plans with a qualified doctor or dietitian."
    )

    if st.button("Save Profile Details"):
        upsert_profile(
            {
                "updated_at": datetime.now().isoformat(),
                "profile_key": active_profile_key,
                "profile_name": active_profile,
                "age": age,
                "blood_sugar": blood_sugar,
                "budget": budget,
                "festival": festival,
                "preferences": preferences,
            }
        )
        st.success(f"Profile '{active_profile}' saved.")

    generate_clicked = st.button("Generate 7-Day Low-GI Meal Plan", type="primary")

with right:
    st.subheader("30-Day History")
    history_df = profile_history.sort_values("created_at", ascending=False).head(20) if not profile_history.empty else pd.DataFrame()
    if history_df.empty:
        st.caption("No plans generated yet.")
    else:
        preview = history_df[["created_at", "mode", "age", "blood_sugar", "festival", "validator_score", "medically_safe"]].copy()
        st.dataframe(preview, use_container_width=True, hide_index=True, height=220)

    st.subheader("Profile Trends")
    if profile_history.empty:
        st.caption("Generate plans for this profile to see trends.")
    else:
        trend_df = profile_history.sort_values("created_at").copy()
        trend_df["created_at"] = pd.to_datetime(trend_df["created_at"], errors="coerce")
        if "validator_score" in trend_df.columns:
            validator_series = trend_df.set_index("created_at")["validator_score"].dropna()
            if validator_series.shape[0] >= 2:
                st.line_chart(data=validator_series, height=180)
            elif validator_series.shape[0] == 1:
                st.caption("Need at least 2 plans to draw validator score trend.")
        if "avg_gi" in trend_df.columns:
            gi_series = trend_df.set_index("created_at")["avg_gi"].dropna()
            if gi_series.shape[0] >= 2:
                st.line_chart(data=gi_series, height=180)
            elif gi_series.shape[0] == 1:
                st.caption("Need at least 2 plans to draw GI trend.")

if generate_clicked:
    query = build_query(age, blood_sugar, budget, festival, preferences)
    user_age = int(age)
    user_bs = int(blood_sugar)

    rag_text = ""
    baseline_text = ""

    with st.spinner(f"Generating plan in mode: {mode}..."):
        if mode == "RAG":
            rag_text = run_rag(llm, retriever, query)
        elif mode == "Baseline":
            baseline_text = run_baseline(llm, query)
        else:
            rag_text = run_rag(llm, retriever, query)
            baseline_text = run_baseline(llm, query)

    if rag_text:
        rag_text = ensure_complete_grocery_list(llm, rag_text, str(budget), festival, preferences, price_map)
    if baseline_text:
        baseline_text = ensure_complete_grocery_list(llm, baseline_text, str(budget), festival, preferences, price_map)

    rag_validation: dict[str, Any] = {}
    base_validation: dict[str, Any] = {}

    if mode in {"RAG", "Compare Both"} and rag_text:
        rag_validation = validate_plan(rag_text, user_age, user_bs)
        st.subheader("RAG Plan")
        st.markdown(rag_text)
        st.caption(
            f"Validator score: {rag_validation['overall_score']} | "
            f"Avg GI: {rag_validation['avg_gi']} | "
            f"Risk: {rag_validation['risk_flag']}"
        )
        st.caption(
            f"Safety: {'PASS' if rag_validation['medically_safe'] else 'FAIL'} | "
            f"Hard fails: {rag_validation['hard_fail_count']} | "
            f"Rules: {rag_validation['violations']}"
        )

    if mode in {"Baseline", "Compare Both"} and baseline_text:
        base_validation = validate_plan(baseline_text, user_age, user_bs)
        st.subheader("Baseline Plan (No RAG)")
        st.markdown(baseline_text)
        st.caption(
            f"Validator score: {base_validation['overall_score']} | "
            f"Avg GI: {base_validation['avg_gi']} | "
            f"Risk: {base_validation['risk_flag']}"
        )
        st.caption(
            f"Safety: {'PASS' if base_validation['medically_safe'] else 'FAIL'} | "
            f"Hard fails: {base_validation['hard_fail_count']} | "
            f"Rules: {base_validation['violations']}"
        )

    chosen_text = rag_text if rag_text else baseline_text
    chosen_mode = "RAG" if rag_text else "Baseline"
    chosen_validation = validate_plan(chosen_text, user_age, user_bs) if chosen_text else {
        "overall_score": 0.0,
        "avg_gi": 0.0,
        "risk_flag": "unknown",
        "medically_safe": False,
        "hard_fail_count": 0,
        "violations": "none",
        "daily_calorie_estimate": 0.0,
    }

    if mode == "Compare Both" and rag_text and baseline_text:
        rag_score = extract_overall_score(rag_validation)
        base_score = extract_overall_score(base_validation)
        rag_safe = bool(rag_validation.get("medically_safe", False))
        base_safe = bool(base_validation.get("medically_safe", False))

        if rag_safe != base_safe:
            pick_rag = rag_safe
        else:
            pick_rag = rag_score >= base_score

        if pick_rag:
            chosen_text = rag_text
            chosen_mode = "RAG"
            chosen_validation = rag_validation
        else:
            chosen_text = baseline_text
            chosen_mode = "Baseline"
            chosen_validation = base_validation

        st.success(f"Auto-selected stronger output for export: {chosen_mode}")

    if chosen_text and not bool(chosen_validation.get("medically_safe", False)):
        st.warning("Selected plan did not pass strict safety checks. Review violations before use.")

    if chosen_text:
        upsert_profile(
            {
                "updated_at": datetime.now().isoformat(),
                "profile_key": active_profile_key,
                "profile_name": active_profile,
                "age": age,
                "blood_sugar": blood_sugar,
                "budget": budget,
                "festival": festival,
                "preferences": preferences,
            }
        )

        st.caption(
            f"Active profile: {active_profile} | "
            f"Safety: {'PASS' if chosen_validation.get('medically_safe') else 'FAIL'} | "
            f"Rules: {chosen_validation.get('violations', 'none')}"
        )

        try:
            pdf_path = save_pdf(chosen_text)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download PDF", pdf_file, file_name=pdf_path.name)
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

        try:
            if mode == "Compare Both" and rag_text and baseline_text:
                st.caption("RAG Audio Explanation")
                rag_clips = save_voice_clips(rag_text, label="rag")
                for clip in rag_clips:
                    st.audio(str(clip), format="audio/mp3")

                st.caption("Baseline Audio Explanation")
                base_clips = save_voice_clips(baseline_text, label="baseline")
                for clip in base_clips:
                    st.audio(str(clip), format="audio/mp3")
            else:
                voice_clips = save_voice_clips(chosen_text, label=chosen_mode.lower())
                for clip in voice_clips:
                    st.audio(str(clip), format="audio/mp3")
        except Exception as exc:
            st.warning(f"Voice generation failed: {exc}")

        append_history(
            {
                "created_at": datetime.now().isoformat(),
                "profile_key": active_profile_key,
                "profile_name": active_profile,
                "mode": chosen_mode,
                "age": age,
                "blood_sugar": blood_sugar,
                "budget": budget,
                "festival": festival,
                "preferences": preferences,
                "validator_score": chosen_validation["overall_score"],
                "avg_gi": chosen_validation["avg_gi"],
                "daily_calorie_estimate": chosen_validation.get("daily_calorie_estimate", 0),
                "violations": chosen_validation.get("violations", "none"),
                "hard_fail_count": chosen_validation.get("hard_fail_count", 0),
                "medically_safe": chosen_validation.get("medically_safe", False),
                "risk_flag": chosen_validation["risk_flag"],
            }
        )
        st.success("Meal plan generated, strictly validated, and saved to 30-day profile history.")
