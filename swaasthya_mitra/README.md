# SwaasthyaMitra

SwaasthyaMitra is a Streamlit plus RAG application that generates low-GI diabetic meal plans for Mumbai families using local food context, strict safety validation, and profile-based history tracking.

Implementation plan and status are maintained in IMPLEMENTATION_PHASES.md.

## Core capabilities

- RAG and baseline generation modes
- Compare mode with automatic winner selection
- Strict validator with violation reporting and hard-fail checks
- Profile-based 30-day history and trend charts
- PDF export and Hindi audio explanations
- Evaluation pipeline with quantitative charts and summary metrics

## Project structure

```text
swaasthya_mitra/
|-- app.py
|-- requirements.txt
|-- .env
|-- .gitignore
|-- Dockerfile
|-- IMPLEMENTATION_PHASES.md
|-- data/
|   |-- pdfs/
|   |-- csvs/
|   `-- chroma_db/
|-- outputs/
|-- evaluation/
|   |-- test_cases.csv
|   |-- run_evaluation.py
|   `-- charts.py
`-- scripts/
	|-- phase1_smoke_test.py
	|-- phase2_validate.py
	|-- phase3_validator_check.py
	|-- phase4_profile_check.py
	`-- rag_chatbot_check.py
```

## Setup and run

### 1) Create environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional one-shot setup:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_phase1.ps1
```

### 2) Configure API key

Edit .env and set your Groq key:

```env
GROQ_API_KEY=your_real_groq_api_key
```

### 3) Add data

Required files:

- data/pdfs with one or more guideline PDFs
- data/csvs/indian_nutrition_2025.csv
- data/csvs/indian_recipes_2026.csv

Optional file for budget pricing quality:

- data/csvs/mumbai_prices.csv with columns item and price_per_kg

### 4) Smoke test

```powershell
python scripts/phase1_smoke_test.py
```

### 5) Start app

```powershell
streamlit run app.py --server.port 8501
```

Open http://localhost:8501.

## In-app usage

- RAG: retrieval-grounded meal plan generation
- Baseline: no retrieval, direct model response
- Compare Both: generates both and selects stronger output via safety and score logic

Generated outputs are written under outputs, including profile history and exported assets.

## Phase 5 runbook: evaluation and ISE-2 evidence

Phase 5 goal is to produce measurable model evidence for the report and demo.

### Inputs

- Expand evaluation/test_cases.csv to at least 50 rows
- Include both baseline and rag case_type rows with matching case_id where possible

### Commands

Run evaluator:

```powershell
python evaluation/run_evaluation.py
```

Generate charts and summary artifacts:

```powershell
python evaluation/charts.py
```

### Expected outputs

- evaluation/test_results.csv
- evaluation/metric_summary.csv
- evaluation/accuracy_comparison.png
- evaluation/adherence_comparison.png
- evaluation/safety_comparison.png
- evaluation/latency_comparison.png
- evaluation/hard_fail_comparison.png when strict safety fields are present
- evaluation/medical_safety_pass_rate.png when strict safety fields are present
- evaluation/paired_accuracy_delta.csv when baseline and rag rows share case_id

### Phase 5 definition-of-done checklist

- Evaluator runs without error
- Charts are generated successfully
- Summary CSV is present and interpretable
- Paired delta output is produced for matched cases

## Phase 6 runbook: demo hardening and submission pack

Phase 6 goal is a reliable 5-minute demonstration from clean restart.

### Pre-demo hardening tasks

- Run two to three end-to-end profile scenarios
- Pre-generate one sample PDF and one sample audio explanation
- Verify safety warnings show correctly for unsafe plans
- Verify profile history isolation and trend charts

### Suggested demo sequence

1. Launch app and show profile selector
2. Generate plan in Compare mode
3. Show selected output, safety result, and violations
4. Show complete grocery list with estimated weekly total
5. Download PDF and play audio explanation
6. Show profile history and trend charts

### Submission pack checklist

- Final README
- IMPLEMENTATION_PHASES.md status up to date
- evaluation artifacts generated
- requirements.txt frozen
- data and outputs backed up

### Optional Docker verification

```powershell
docker build -t swaasthya-mitra .
docker run -p 8501:8501 -v "${PWD}/data:/app/data" swaasthya-mitra
```

## Troubleshooting

- If streamlit run app.py fails with file not found, run command from swaasthya_mitra folder or use absolute path.
- If port 8501 is busy, stop existing process or choose a different port.
- If startup warns about missing data files, add required CSV and PDF files under data.
