# SwaasthyaMitra

SwaasthyaMitra is a Streamlit + RAG application for generating culturally relevant, low-GI diabetic meal plans for Mumbai families.

Implementation phases are documented in [IMPLEMENTATION_PHASES.md](IMPLEMENTATION_PHASES.md).

Implemented features:

- RAG meal planning from PDFs + Indian nutrition CSVs
- Baseline (no retrieval) vs RAG comparison mode
- Strict validator with hard safety rules and explicit violations
- 30-day persistent generation history with family profiles
- Trend charts per profile (validator score and avg GI)
- PDF export and Hindi MP3 output
- ISE-2 evaluation pipeline with metric charts

## Folder structure

```
swaasthya_mitra/
|-- data/
|   |-- pdfs/
|   |-- csvs/
|   `-- chroma_db/
|-- outputs/
|-- evaluation/
|   |-- test_cases.csv
|   `-- charts.py
|-- .env
|-- .gitignore
|-- app.py
|-- Dockerfile
`-- requirements.txt
```

## 1. Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Quick Phase 1 setup (recommended):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_phase1.ps1
```

Manual smoke test:

```powershell
python scripts/phase1_smoke_test.py
```

## 2. Configure API key

Edit `.env`:

```env
GROQ_API_KEY=your_groq_key_here
```

## 3. Add data files

Place files at:

- `data/pdfs/`: one or more PDF guidelines/charts
- `data/csvs/indian_nutrition_2025.csv`
- `data/csvs/indian_recipes_2026.csv`

## 4. Run app

```powershell
streamlit run app.py
```

Open: http://localhost:8501

### In-app modes

- `RAG`: Uses vector retrieval for evidence-grounded plans
- `Baseline`: Uses pure LLM response without retrieval
- `Compare Both`: Generates both and auto-picks stronger output using validator score

### 30-day memory tracking

- Every generation is saved to `outputs/plan_history.csv`
- Old records beyond 30 days are automatically removed
- Recent profile-specific history is visible in the right panel of the app

### Strict safety validator

- Checks for GI availability and high-GI items
- Checks meal coverage completeness
- Checks calorie range against fasting blood sugar category
- Flags direct sugar terms (e.g. sugar/jaggery)
- Stores hard-fail count and medical pass/fail in history

## 5. Optional Docker run

```powershell
docker build -t swaasthya-mitra .
docker run -p 8501:8501 -v "${PWD}/data:/app/data" swaasthya-mitra
```

## 6. Evaluation pipeline (ISE-2)

Prepare or edit `evaluation/test_cases.csv`.

Run model evaluation (creates `evaluation/test_results.csv`):

```powershell
python evaluation/run_evaluation.py
```

Generate charts and summary table from the results CSV:

```powershell
python evaluation/charts.py
```

Produced files:

- `evaluation/accuracy_comparison.png`
- `evaluation/adherence_comparison.png`
- `evaluation/safety_comparison.png`
- `evaluation/latency_comparison.png`
- `evaluation/hard_fail_comparison.png` (if strict fields present)
- `evaluation/medical_safety_pass_rate.png` (if strict fields present)
- `evaluation/metric_summary.csv`

If your `test_results.csv` has both `baseline` and `rag` entries with same `case_id`, a paired delta file is also generated:

- `evaluation/paired_accuracy_delta.csv`
