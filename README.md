# IGNIS NOVA · Flame in Freefall

SI-powered dashboard for the NASA Space Apps Challenge 2026: it ranks, compares and interprets
microgravity combustion investigations for spacecraft, Moon-base and rover fire safety.

## Files
- `app.py` — Streamlit interface (7 tabs)
- `ignis_core.py` — data, scoring, evidence-readiness index, gap analysis, SI engines
- `requirements.txt` — dependencies
- `.streamlit/config.toml` — dark theme (put it in a folder named `.streamlit`)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repository and upload all four files (keep `.streamlit/config.toml` in its folder).
2. Go to https://share.streamlit.io, click **New app**, pick the repository, main file `app.py`.
3. Optional LLM answers: in the app's **Settings → Secrets**, add
   `ANTHROPIC_API_KEY = "your-key"`. Without it, the SI Analyst uses the built-in local engine.

## Plug in real PSI data
Download the CSV template from the **Method & Data** tab, fill one row per investigation
(ratings 0-5), upload it in the sidebar.
