import logging
import re
import sys
import time
from pathlib import Path

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

load_dotenv()

t0 = time.time()
root = Path('.')
chroma_dir = root / 'data' / 'chroma_db'

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vs = Chroma(
	embedding_function=emb,
	persist_directory=str(chroma_dir),
	client_settings=Settings(anonymized_telemetry=False),
)

llm = ChatGroq(model_name='llama-3.3-70b-versatile', temperature=0.3, max_tokens=1200)
query = 'Create a detailed 7-day low-GI diabetic meal plan for age 45 and fasting sugar 150 mg/dL in Mumbai with affordable ingredients.'


def has_complete_7_day_structure(text: str) -> bool:
	day_headers = set(
		int(match.group(1))
		for match in re.finditer(r"\bday\s*([1-7])\b", text, flags=re.IGNORECASE)
	)
	return day_headers == {1, 2, 3, 4, 5, 6, 7}


def enforce_7_day_structure(text: str) -> str:
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

qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vs.as_retriever(search_kwargs={'k': 6}))
rag = qa.invoke({'query': query + '\n\nUse retrieved evidence when available. If evidence is partial, still provide a complete 7-day plan with headings Day 1 to Day 7 and clearly stated assumptions. Do not refuse the request.'}).get('result', '')
rag = enforce_7_day_structure(rag)

base = llm.invoke('Answer directly without any external retrieval. Generate a complete 7-day plan with headings Day 1 to Day 7. Include reasonable assumptions and confidence caveats for uncertain GI values.\n\n' + query).content
base = enforce_7_day_structure(base)

day_re = re.compile(r'\b(day\s*[1-7]|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I)
rag_days = len(set(m.group(0).lower() for m in day_re.finditer(rag)))
base_days = len(set(m.group(0).lower() for m in day_re.finditer(base)))

required_day_headers = {f"day {i}" for i in range(1, 8)}
rag_headers = set(m.group(0).lower().replace("  ", " ").strip() for m in re.finditer(r"\bday\s*[1-7]\b", rag, re.I))
base_headers = set(m.group(0).lower().replace("  ", " ").strip() for m in re.finditer(r"\bday\s*[1-7]\b", base, re.I))

k4_docs = len(vs.as_retriever(search_kwargs={'k': 4}).invoke('low gi plan'))
k6_docs = len(vs.as_retriever(search_kwargs={'k': 6}).invoke('low gi plan'))

print('phase2_check: OK')
print('elapsed_s=', round(time.time() - t0, 2))
print('rag_chars=', len(rag))
print('baseline_chars=', len(base))
print('rag_day_markers=', rag_days)
print('baseline_day_markers=', base_days)
print('k4_docs=', k4_docs)
print('k6_docs=', k6_docs)
print('compare_ready=', bool(rag.strip()) and bool(base.strip()))
print('rag_has_day1_to_day7=', rag_headers == required_day_headers)
print('baseline_has_day1_to_day7=', base_headers == required_day_headers)

checks = [
	bool(rag.strip()),
	bool(base.strip()),
	k4_docs == 4,
	k6_docs == 6,
	rag_headers == required_day_headers,
	base_headers == required_day_headers,
]

if not all(checks):
	print('phase2_check: FAILED')
	sys.exit(1)

print('phase2_check: PASSED_STRICT')
