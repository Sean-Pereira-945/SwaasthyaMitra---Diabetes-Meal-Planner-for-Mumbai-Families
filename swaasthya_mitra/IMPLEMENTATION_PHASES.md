# SwaasthyaMitra Implementation Phases (7-12 Days)

This phased plan is designed for a 3-4 member team and an 8 GB RAM laptop setup.

## Current Execution Status

- Phase 1: Completed
	- Smoke test: pass
	- Environment/dependencies: ready
	- Data folders and key setup: ready
	
- Phase 2: Completed
	- Retriever indexing speed check: completed
	- Retriever `k` behavior (`k=4`, `k=6`): completed
	- Baseline/RAG/Compare runtime validation: passed
	- Strict 7-day output structure validation: passed


- Phase 3: Completed
	- Validator edge-case tests (high GI, missing calories, sugary terms): passed
	- Unsafe-plan warning and violation surfacing in app: active
	- Final validator sign-off: passed

- Phase 4: In Progress
	- Profile isolation check: passed
	- 30-day pruning check: passed

## Phase 1: Foundation and Environment (Day 1)
Goal: Make the project runnable on every teammate machine.

Deliverables:
- Python environment configured
- Dependencies installed
- Data folders in place
- API key configured
- Smoke test passing

Tasks:
- Run setup script: `powershell -ExecutionPolicy Bypass -File scripts/setup_phase1.ps1`
- Add source data files under `data/pdfs` and `data/csvs`
- Update `.env` with valid `GROQ_API_KEY`
- Run smoke test: `python scripts/phase1_smoke_test.py`

Definition of done:
- Smoke test exits with code 0
- Team can run `streamlit run app.py` without syntax/import failures

## Phase 2: Core Meal Planning (Day 2-3)
Goal: Complete baseline and RAG generation for meal plans.

Deliverables:
- Working RAG mode
- Working Baseline mode
- Compare mode with automatic selection

Tasks:
- Validate retriever indexing speed and memory
- Verify answer consistency at `temperature=0.3`
- Tune retriever `k` between 4 and 6 if memory pressure appears

Definition of done:
- Each mode returns complete 7-day plan
- Compare mode auto-selects output and logs choice

## Phase 3: Safety and Explainability (Day 3-4)
Goal: Enforce strict diet safety constraints and transparent failure reasons.

Deliverables:
- Strict validator integrated
- Hard-fail count and pass/fail safety status in outputs
- Rule violation strings saved in history

Tasks:
- Test validator against edge prompts (high GI, missing calories, sugary terms)
- Confirm warnings appear in UI when plan is unsafe

Definition of done:
- Unsafe plans are clearly flagged in app and history

## Phase 4: Memory and Personalization (Day 4-5)
Goal: Support multi-family profile tracking and trends.

Deliverables:
- Profile selector + profile creation
- Profile-specific 30-day history
- Trend charts (validator score, average GI)

Tasks:
- Verify profile isolation in history rows
- Confirm automatic pruning beyond 30 days

Definition of done:
- Multiple profiles can be used without data mixing

## Phase 5: Evaluation and ISE-2 Evidence (Day 5-7)
Goal: Produce measurable comparison artifacts for report/demo.

Deliverables:
- Automated evaluator output CSV
- Metric charts for accuracy, adherence, safety, latency
- Strict safety charts (hard fails, medical pass rate)

Tasks:
- Expand `evaluation/test_cases.csv` to 50+ rows
- Run evaluation and generate all charts
- Validate paired baseline-vs-rag delta output

Definition of done:
- All evaluation charts and summary CSV generated without errors

## Phase 6: Demo Hardening and Submission Pack (Day 7-12)
Goal: Make a stable submission-ready prototype.

Deliverables:
- Final README instructions
- Stable runbook for demo day
- Optional Docker image verified

Tasks:
- Run end-to-end demo scenario for 2-3 profiles
- Pre-generate one sample output PDF and audio
- Freeze dependency versions and back up data files

Definition of done:
- Team can demo in under 5 minutes from clean restart

## Team Split Recommendation
- Member 1: Core LLM/RAG + performance tuning
- Member 2: UI + profile memory + exports
- Member 3: Evaluation pipeline + charts + ISE report evidence
- Member 4 (optional): Data cleaning + prompt refinement + QA
