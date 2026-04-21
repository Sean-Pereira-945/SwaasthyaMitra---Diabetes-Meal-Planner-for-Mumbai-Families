# SwaasthyaMitra Diabetes-Meal-Planner-for-Mumbai-Families

Main project files are in the swaasthya_mitra folder.

See the full project guide here:
- swaasthya_mitra/README.md

## Streamlit deployment guardrails

- Keep root requirements.txt in place. It forwards installs to swaasthya_mitra/requirements.txt so Streamlit Cloud picks up all dependencies.
- Run this before push and deploy:

```powershell
python swaasthya_mitra/scripts/predeploy_check.py
```

- In Streamlit Cloud, set app entrypoint to swaasthya_mitra/app.py.
