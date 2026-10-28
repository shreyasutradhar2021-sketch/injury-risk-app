# AI-Assisted Player Health & Injury Risk System

Team 8 · Section D · Foundations of AI for Business · Project 4 (Healthcare)

> **Educational prototype. Not medical advice.** Every decision is made by the club's medical team.

Staff enter each player's weekly training load and wellness. A **predictive model** rates next-week injury
risk Low or High with its top three reasons. A **RAG component** retrieves the club's own protocol
clauses and an LLM (Groq) drafts a briefing that cites them. **Guardrails** check the input and the
output. Nothing reaches the head coach until the **club doctor or head physio signs off**.

```
weekly data ──► input checks ──► risk model ──► protocol retrieval ──► LLM briefing ──► output checks ──► MEDICAL SIGN-OFF ──► coach view
 (sports        (ranges,          (logistic       (rule-matched +        (Groq, JSON,       (clauses, numbers,    (approve / modify /
  scientist)     injection)        regression)     question search)       temp 0)            banned phrases)       override + reason)
                                                                          └─ fails twice → fixed template          everything → audit log
```

## Run it locally (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your Groq key into it
streamlit run app.py
```

Without a Groq key the app still works: briefings use the fixed fallback template.

## Deploy (public link for demo day)

1. Push this folder to a **public GitHub repo** (`.gitignore` already keeps keys and the audit DB out).
2. On [share.streamlit.io](https://share.streamlit.io) → *Create app* → pick the repo, branch, `app.py`, Python **3.12**.
3. *Advanced settings → Secrets*: paste
   ```
   groq_api_key = "gsk_..."
   staff_pin = "choose-a-pin"      # opens the medical views (coach view needs none)
   signoff_pin = "another-pin"     # club doctor / head physio sign-off
   ```
4. Open the link once before the demo (free apps sleep after inactivity).

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit app: squad dashboard, briefing + what-if + sign-off, data entry, coach view, evaluation, audit log, protocol |
| `train_model.py` | Builds features, trains and evaluates the model, writes `model/`, `eval/model_metrics.json`, `data/demo_squad.csv` |
| `features.py` | Feature maths shared by training and the app; input ranges |
| `risk.py` | Scoring, top-3 reasons, what-if; retrains automatically if the saved model can't load |
| `protocol.py` + `knowledge/protocol.md` | 17-clause club protocol, read fresh every run; rule-based + TF-IDF retrieval |
| `briefing.py` | Groq call (JSON mode, temperature 0, retry/backoff, model fallback), Pydantic schema, re-prompt once, fallback template |
| `guardrails.py` | Injection screen (keywords + Llama Prompt Guard 2 on Groq), output validation |
| `audit.py` | SQLite audit log |
| `eval/run_evaluation.py` | Retrieval, red-team and output-check tests; end-to-end LLM tests when a key is present |

## Results so far

**Data (real, not synthetic):** Lövdal, den Hartigh & Azzopardi (2021), *Injury Prediction in Competitive Runners
With Machine Learning*, CC0. 74 elite runners, 7 years, 42,798 athlete-days, 575 injuries.
Tested on **15 athletes the model never saw**.

| Model (held-out athletes) | AUC | Flagged | Injuries caught |
|---|---|---|---|
| Always "Low risk" | 0.50 | 0% | 0% |
| Flag if ACWR ≥ 1.5 | 0.53 | 10% | 15% |
| Flag if injured in last 12 months | 0.67 | 48% | 81% |
| **Logistic regression, chosen cut-off** | **0.70** | **16%** | **31%** |
| Gradient boosting (comparison) | 0.70 | 18% | 41% |

5-fold cross-validation by athlete: AUC 0.665 ± 0.009. Squad projection (30 players, 2 injuries per player
per season): about 5 players flagged per week, 19 of 60 injuries caught, weekly precision about 9%.

**LLM & guardrails:** retrieval 25/25 (20 covered + 5 "not covered"); input screen blocks 17/20 attacks with
0/10 normal notes wrongly blocked; output checks catch 17/17 planted bad briefings (the original v1 checks: 11/17).
End-to-end on Groq the requested Llama models were no longer available to our key, so answers came from the
fallback **openai/gpt-oss-120b** (the app now detects available models and reports the one that answered). It passed
25/25 questions and broke no rule on 20 attacks, so a 15-case harder attack set was added
(`python eval/run_evaluation.py --hard`, and `--hard --model small` for the small-model comparison).

## Changes since the interim report

| Interim plan | Now | Why |
|---|---|---|
| Synthetic squad data | Real runners' dataset as the main data | Professor asked for real data |
| Gemini free tier | Groq; model picked from what the key can use (now openai/gpt-oss-120b, small: gpt-oss-20b) | Course setup guide and sample project use Groq; Llama 3.3/3.1 were retired for our key |
| MiniLM embeddings + FAISS | Rule-matched retrieval + TF-IDF | 17 short clauses; fits Streamlit's 1 GB memory; retrieval is testable |
| 19 features incl. sleep | 6 features; sleep handled by protocol clause C-8 | Dataset has no sleep; simpler model scored the same; 'training success' dropped because 19 athletes never logged it |
| Cut-off for ~6 reviews/week | Cut-off that catches 50% of training injuries | Missed injuries cost far more than reviews |
