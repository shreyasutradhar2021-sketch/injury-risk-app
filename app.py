"""
AI-Assisted Player Health & Injury Risk System - Team 8, Project 4 (Healthcare)
Educational prototype. Not medical advice.

Run locally:  streamlit run app.py
"""

import hashlib
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import audit
from briefing import BACKUP_MODEL, DEFAULT_MODEL, generate_briefing
from features import INPUT_FIELDS
from guardrails import screen_text
from protocol import NOT_COVERED, load_protocol, retrieve
from risk import facts_for_protocol, load_model, score, validate_inputs, what_if
from secrets_util import get_groq_client

ROOT = os.path.dirname(os.path.abspath(__file__))
SQUAD_PATH = os.path.join(ROOT, "data", "demo_squad.csv")
METRICS_PATH = os.path.join(ROOT, "eval", "model_metrics.json")
EVAL_PATH = os.path.join(ROOT, "eval", "evaluation_results.json")
TEST_PRED_PATH = os.path.join(ROOT, "eval", "test_predictions.csv")

STATUS_OPTIONS = [
    "Train as planned",
    "Modified load (medical plan)",
    "Medical assessment before next session",
]
SIGNOFF_ROLES = ("Club doctor", "Head physiotherapist")
ROLES = ["Sports scientist", "Club doctor", "Head physiotherapist", "Head coach"]
DISCLAIMER = ("Educational prototype built for a university course. Not medical advice. "
              "Risk scores come from a model trained on real data from elite runners, not footballers, "
              "and every decision is made by the club's medical team.")

st.set_page_config(page_title="Player Health & Injury Risk", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #f8f7f4; color: #1e293b; }
.hero { background: linear-gradient(135deg, #0f1f3d 0%, #1e3a68 100%); color: #fff; padding: 20px 28px;
        border-radius: 12px; margin-bottom: 14px; }
.hero h1 { font-size: 1.6rem; margin: 0 0 4px 0; color: #fff; letter-spacing: -0.01em; }
.hero p { margin: 0; color: #cbd5e1; font-size: 0.92rem; }
.disclaimer { background: #fff8e1; border: 1px solid #f3d27a; color: #6b4e00; padding: 8px 14px;
              border-radius: 8px; font-size: 0.85rem; margin-bottom: 16px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; }
.card .label { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; }
.card .value { font-size: 1.7rem; font-weight: 700; color: #0f1f3d; line-height: 1.15; }
.card .sub { font-size: 0.75rem; color: #94a3b8; }
.mono { font-family: 'DM Mono', monospace; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; }
.pill-high { background: #ffebee; color: #b71c1c; }
.pill-low { background: #e8f5e9; color: #1b5e20; }
.pill-border { background: #fff8e1; color: #b26a00; }
.pill-gen { background: #e3f2fd; color: #0d47a1; }
.clause { background: #fff; border-left: 4px solid #1e3a68; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", color="#0f1f3d"):
    st.markdown(f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ start-up checks
missing = [p for p in [SQUAD_PATH, os.path.join(ROOT, "knowledge", "protocol.md"),
                       os.path.join(ROOT, "data", "week_approach_maskedID_timeseries.csv")]
           if not os.path.exists(p)]
if missing:
    st.error("⚠️ Missing required file(s): " + ", ".join(os.path.relpath(m, ROOT) for m in missing))
    st.stop()


@st.cache_resource(show_spinner="Loading the risk model...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner=False)
def get_client():
    return get_groq_client()


try:
    MODEL = get_model()
except Exception as e:  # noqa: BLE001
    st.error(f"⚠️ Could not load or train the risk model: {str(e)[:200]}")
    st.stop()

PROTOCOL = load_protocol()          # read fresh on every run, like the professor's rules file
BAND = PROTOCOL.setting("borderline_band", 0.10)
CLIENT = get_client()


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ session state
ss = st.session_state
if "squad" not in ss:
    ss.squad = pd.read_csv(SQUAD_PATH)
if "briefings" not in ss:
    ss.briefings = {}      # cache: key -> briefing result (so reruns never call the LLM again)
if "signoffs" not in ss:
    ss.signoffs = {}       # player_id -> sign-off record
if "notes" not in ss:
    ss.notes = {}


def rec_hash(rec):
    keys = [k for k, *_ in INPUT_FIELDS]
    return hashlib.md5(json.dumps({k: float(rec[k]) for k in keys}, sort_keys=True).encode()).hexdigest()[:10]


def score_squad(squad):
    rows = []
    for rec in squad.to_dict("records"):
        res = score(MODEL, rec, BAND)
        rows.append((rec, res))
    return rows


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚽ Medical Room")
    st.caption("Team 8 · Project 4 (Healthcare)")
    role = st.selectbox("I am the...", ROLES, index=0,
                        help="Views change with the role. Only the club doctor or head physio can sign off; "
                             "the head coach only sees signed-off outcomes (protocol C-16).")
    reviewer = st.text_input("Your name (for the audit log)", value="Demo user")
    st.markdown("---")
    st.markdown("### 🤖 Briefing LLM (Groq)")
    if CLIENT is None:
        st.warning("No Groq API key found. Briefings use the fixed fallback template.")
    else:
        st.success("Groq key loaded from secrets.")
    llm_model = st.selectbox("Model", [DEFAULT_MODEL, BACKUP_MODEL], index=0)
    st.markdown("---")
    if st.button("🔄 Reset demo (squad, briefings, sign-offs)", use_container_width=True):
        for k in ["squad", "briefings", "signoffs", "notes"]:
            ss.pop(k, None)
        audit.log_event("demo_reset", actor=reviewer)
        st.rerun()
    st.caption(f"Cut-off: {MODEL['threshold']:.4f} · borderline band ±{BAND:.0%}")

# ------------------------------------------------------------------ header
st.markdown('<div class="hero"><h1>AI-Assisted Player Health & Injury Risk</h1>'
            '<p>Weekly load & wellness → risk model → protocol-grounded briefing → medical sign-off → coach</p></div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

scored = score_squad(ss.squad)
player_ids = list(ss.squad["player_id"])


def signoff_label(pid, rec):
    s = ss.signoffs.get(pid)
    if not s:
        return "⏳ Awaiting sign-off"
    if s["data_hash"] != rec_hash(rec):
        return "⏳ New data - re-sign"
    return f"✅ {s['final_status']}"


# ================================================================== HEAD COACH VIEW
if role == "Head coach":
    st.subheader("Coach view: signed-off player status")
    st.caption("You see only outcomes signed off by the medical team. Raw wellness data and risk scores "
               "are not shown to coaching staff (protocol C-16).")
    rows = []
    for rec, res in scored:
        pid = rec["player_id"]
        s = ss.signoffs.get(pid)
        if s and s["data_hash"] == rec_hash(rec):
            rows.append({"Player": pid, "Status": s["final_status"], "Medical note": s["coach_note"],
                         "Signed off by": f"{s['by']} ({s['role']})", "Signed at": s["at"]})
        else:
            rows.append({"Player": pid, "Status": "Awaiting medical sign-off", "Medical note": "—",
                         "Signed off by": "—", "Signed at": "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=600)
    st.stop()

# ================================================================== STAFF VIEWS
tabs = st.tabs(["📋 Squad dashboard", "🩺 Player briefing & sign-off", "✍️ Weekly data entry",
                "👀 Coach view", "📊 Evaluation", "🗂️ Audit log", "📜 Protocol"])

# ------------------------------------------------------------------ 1. squad dashboard
with tabs[0]:
    n_high = sum(r["risk_level"] == "High" for _, r in scored)
    n_border = sum(r["borderline"] for _, r in scored)
    n_signed = sum(signoff_label(rec["player_id"], rec).startswith("✅") for rec, _ in scored)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("Players", len(scored), "this week's entries")
    with c2:
        card("High risk", n_high, "model flag", "#b71c1c")
    with c3:
        card("Borderline", n_border, f"within ±{BAND:.0%} of cut-off", "#b26a00")
    with c4:
        card("Awaiting sign-off", len(scored) - n_signed, "medical team")
    with c5:
        card("Signed off", n_signed, "visible to coach", "#1b5e20")
    st.write("")

    rows = []
    for rec, res in scored:
        lvl = res["risk_level"]
        tag = ("🔴 High" if lvl == "High" else "🟢 Low") + (" · borderline" if res["borderline"] else "")
        facts = facts_for_protocol(rec, res)
        _, info = retrieve(PROTOCOL, dict(facts))
        rows.append({
            "Player": rec["player_id"],
            "Risk": tag,
            "Risk index": round(res["risk_index"], 1),
            "Main risk factor": res["reasons"][0] if res["reasons"] else "—",
            "ACWR": round(res["features"]["acwr"], 2),
            "Distance (km)": rec["km_this_week"],
            "Sleep (h)": rec["sleep_hours"],
            "Protocol flags": ", ".join(c for c in info["rule_clauses"] if c not in ("C-1", "C-2", "C-3", "C-16")) or "—",
            "Conflict": "⚠️ " + ", ".join(f"{a}/{b}" for a, b in info["conflict_pairs"]) if info["conflict_pairs"] else "",
            "Sign-off": signoff_label(rec["player_id"], rec),
        })
    df = pd.DataFrame(rows).sort_values("Risk index", ascending=False)

    def color_risk(v):
        v = str(v)
        if "High" in v:
            return "background-color:#ffebee;color:#b71c1c;font-weight:600"
        if "borderline" in v:
            return "background-color:#fff8e1;color:#b26a00;font-weight:600"
        if "Low" in v:
            return "background-color:#e8f5e9;color:#1b5e20;font-weight:600"
        return ""

    styled = df.style.map(color_risk, subset=["Risk"]).format(
        {"Risk index": "{:.1f}", "ACWR": "{:.2f}", "Distance (km)": "{:.1f}", "Sleep (h)": "{:.1f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=560,
                 column_config={
                     "Risk index": st.column_config.NumberColumn(help="Model risk ÷ average player-week. 2.0 = twice the average."),
                     "ACWR": st.column_config.NumberColumn(help="This week's distance ÷ average of the last 3 weeks"),
                     "Protocol flags": st.column_config.TextColumn(help="Extra protocol clauses triggered by this player's numbers"),
                 })
    st.caption("Squad = 30 real athlete-weeks from the held-out test athletes (anonymised as P-01…P-30). "
               "Sleep values are illustrative: the dataset has no sleep data, so sleep only drives protocol clause C-8, not the model.")
    st.download_button("📥 Download squad table (CSV)", df.to_csv(index=False), "squad_risk.csv", "text/csv")

# ------------------------------------------------------------------ 2. player briefing
with tabs[1]:
    order = [r["player_id"] for r, _ in sorted(scored, key=lambda t: -t[1]["probability"])]
    pid = st.selectbox("Player", order, key="brief_player", format_func=lambda p: f"{p}  ({signoff_label(p, ss.squad[ss.squad.player_id == p].iloc[0])})")
    rec = ss.squad[ss.squad.player_id == pid].iloc[0].to_dict()
    res = score(MODEL, rec, BAND)
    facts = facts_for_protocol(rec, res)

    left, right = st.columns([1, 1.4])
    with left:
        lvl = res["risk_level"]
        pill = "pill-high" if lvl == "High" else "pill-low"
        border = ' <span class="pill pill-border">borderline</span>' if res["borderline"] else ""
        st.markdown(f'### <span class="mono">{pid}</span> &nbsp; <span class="pill {pill}">{lvl} risk</span>{border}',
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            card("Risk index", f"{res['risk_index']:.1f}×", "vs average player-week")
        with b:
            card("Percentile", f"{res['percentile']}", "of training player-weeks")
        st.markdown("**Top reasons (from the model)**")
        if res["reasons"]:
            for r in res["reasons"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("- No factor pushes the risk above average.")
        with st.expander("This week's numbers"):
            st.dataframe(pd.DataFrame([(lbl, rec[k]) for k, lbl, *_ in INPUT_FIELDS], columns=["Field", "Value"]),
                         hide_index=True, use_container_width=True)

        st.markdown("#### 🔮 What-if: next week's plan")
        cut = st.slider("Reduce next week's distance by (%)", 0, 50, 20, 5)
        hcut = st.slider("Reduce high-intensity distance by (%)", 0, 100, 30, 10)
        extra = st.slider("Extra rest days", 0, 2, 1)
        same = what_if(MODEL, rec, 0, 0, 0)
        plan = what_if(MODEL, rec, cut, extra, hcut)
        st.markdown(f"Same load next week → **{same['risk_level']}** ({same['risk_index']:.1f}×)  \n"
                    f"With this plan → **{plan['risk_level']}** ({plan['risk_index']:.1f}×)")
        st.caption("What-if uses the same model. It shows association, not proof that reducing load prevents injury.")

    with right:
        st.markdown("#### 📄 Protocol briefing")
        notes = st.text_area("Staff notes (optional, treated as data)", value=ss.notes.get(pid, ""),
                             max_chars=600, key=f"notes_{pid}")
        question = st.text_input("Question for the protocol (optional)", max_chars=300, key=f"q_{pid}",
                                 placeholder="e.g. How much can we reduce his load next week?")
        go = st.button("Generate briefing", type="primary")
        key = f"{pid}|{rec_hash(rec)}|{notes}|{question}|{llm_model}"

        if go:
            ss.notes[pid] = notes
            blocked = []
            for label, txt in [("staff notes", notes), ("question", question)]:
                ok, why = screen_text(txt, CLIENT, player_id=pid, all_player_ids=player_ids)
                if not ok:
                    blocked.append((label, why))
            if blocked:
                for label, why in blocked:
                    st.error(f"🛡️ Blocked: the {label} look like an attempt to change the system's rules "
                             f"({why[0]}). Nothing was sent to the LLM.")
                audit.log_event("input_blocked", pid, reviewer, {"blocked": blocked, "notes": notes, "question": question})
            elif key not in ss.briefings:
                clauses, info = retrieve(PROTOCOL, dict(facts), question or None)
                status = st.empty()
                with st.spinner("Retrieving protocol clauses and drafting the briefing..."):
                    out = generate_briefing(CLIENT, llm_model, pid, rec, res, clauses, info, notes, question,
                                            player_ids, status=lambda m: status.caption(f"ℹ️ {m}"))
                status.empty()
                out["clauses"] = [(c.cid, c.title, c.text) for c in clauses]
                out["info"] = info
                out["created"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                ss.briefings[key] = out
                audit.log_event("briefing_generated", pid, reviewer, {
                    "generated_by": out["generated_by"], "attempts": out["attempts"],
                    "guardrail_problems": out["problems"], "clauses": [c[0] for c in out["clauses"]],
                    "risk_level": res["risk_level"], "risk_index": round(res["risk_index"], 2),
                    "question": question, "notes": notes})

        out = ss.briefings.get(key)
        if out is None:
            # show the latest briefing for this player and data, if any
            prev = [v for k, v in ss.briefings.items() if k.startswith(f"{pid}|{rec_hash(rec)}|")]
            out = prev[-1] if prev else None
        if out:
            b = out["briefing"]
            st.markdown(f'<span class="pill pill-gen">{out["generated_by"]}</span> '
                        f'<span style="color:#64748b;font-size:0.8rem">attempts: {out["attempts"]} · {out["created"]}</span>',
                        unsafe_allow_html=True)
            st.markdown(f"**Summary.** {b['summary']}")
            if b.get("key_factors"):
                st.markdown("**Key factors:** " + "; ".join(b["key_factors"]))
            st.markdown("**What the protocol says**")
            for a in b["protocol_actions"]:
                st.markdown(f'<div class="clause"><b class="mono">{a["clause"]}</b> — {a["action"]}</div>',
                            unsafe_allow_html=True)
            if b.get("conflicts"):
                st.warning("**Conflicting clauses:** " + " ".join(b["conflicts"]))
            if b.get("answer_to_question"):
                (st.info if NOT_COVERED.lower()[:20] in b["answer_to_question"].lower() else st.success)(
                    f"**Answer to the question:** {b['answer_to_question']}")
            st.markdown(f"**For the medical team:** {b['for_medical_team']}")
            with st.expander("Retrieved clauses & guardrail log"):
                st.markdown(f"Rule-matched: {', '.join(out['info']['rule_clauses'])}  \n"
                            f"From the question: {', '.join(out['info']['question_clauses']) or '—'} "
                            f"(similarity {out['info']['question_similarities'] or '—'})")
                for cid, title, text in out["clauses"]:
                    st.markdown(f"- **{cid} {title}** — {text}")
                if out["problems"]:
                    st.markdown("**Guardrail checks that failed along the way:**")
                    for p in out["problems"]:
                        st.markdown(f"- {p}")
                else:
                    st.markdown("All output checks passed on the first attempt.")
        else:
            st.info("Press **Generate briefing** to retrieve the relevant clauses and draft a briefing.")

        # ---- human checkpoint ----
        st.markdown("---")
        st.markdown("#### ✅ Medical sign-off")
        current = ss.signoffs.get(pid)
        if current and current["data_hash"] == rec_hash(rec):
            st.success(f"Signed off by {current['by']} ({current['role']}) at {current['at']}: "
                       f"**{current['final_status']}** — decision: {current['decision']}")
        if role not in SIGNOFF_ROLES:
            st.info("Only the club doctor or head physiotherapist can sign off. Switch role in the sidebar.")
        elif out is None:
            st.info("Generate the briefing first, so the sign-off is based on it.")
        else:
            with st.form(f"signoff_{pid}"):
                decision = st.radio("Decision on the briefing", ["Approve", "Modify", "Override"], horizontal=True,
                                    help="Approve = accept as drafted. Modify = change the plan. "
                                         "Override = disagree with the model's risk level.")
                # No pre-selected status, so the reviewer makes an active choice (reduces automation bias).
                final_status = st.selectbox("Training status for this week (your decision)", STATUS_OPTIONS,
                                            index=None, placeholder="Choose a status...")
                coach_note = st.text_area("Note the coach will see (no wellness scores or diagnoses, C-16)",
                                          max_chars=400, placeholder="e.g. Modified load this week; available for selection decisions on Friday.")
                reason = st.text_area("Reason (required for Modify or Override)", max_chars=400)
                submitted = st.form_submit_button("Sign off", type="primary")
            if submitted:
                if final_status is None or not coach_note.strip():
                    st.error("Choose a training status and write the note for the coach.")
                elif decision in ("Modify", "Override") and len(reason.strip()) < 10:
                    st.error("Please give a reason (at least 10 characters) for a Modify or Override.")
                else:
                    # Human-written note: only check it does not name other players (C-16).
                    ok, why = screen_text(coach_note, None, player_id=pid, all_player_ids=player_ids, patterns=[])
                    if not ok:
                        st.error(f"The coach note was not saved: it {why[0]}.")
                    else:
                        ss.signoffs[pid] = {"decision": decision, "final_status": final_status, "coach_note": coach_note,
                                            "reason": reason, "by": reviewer, "role": role,
                                            "at": datetime.now().strftime("%Y-%m-%d %H:%M"), "data_hash": rec_hash(rec),
                                            "model_level": res["risk_level"], "briefing_by": out["generated_by"]}
                        audit.log_event("signoff", pid, f"{reviewer} ({role})", ss.signoffs[pid])
                        st.rerun()

# ------------------------------------------------------------------ 3. data entry
with tabs[2]:
    st.subheader("Weekly data entry")
    st.caption("Every field is range-checked before scoring (input guardrail). New data for a player resets "
               "their sign-off, so stale approvals never reach the coach.")
    mode = st.radio("Entry mode", ["One player (form)", "Whole squad (CSV upload)"], horizontal=True)
    if mode == "One player (form)":
        existing = ["➕ New player"] + player_ids
        who = st.selectbox("Player to add or update", existing, key="entry_player")
        base = ss.squad[ss.squad.player_id == who].iloc[0].to_dict() if who != "➕ New player" else {}
        with st.form("entry"):
            new_id = st.text_input("Player ID", value=who if who != "➕ New player" else f"P-{len(player_ids) + 1:02d}")
            cols = st.columns(3)
            vals = {}
            for i, (k, label, lo, hi, default, step) in enumerate(INPUT_FIELDS):
                with cols[i % 3]:
                    v = base.get(k, default)
                    if isinstance(step, int):
                        vals[k] = st.number_input(label, min_value=None, max_value=None, value=int(v), step=step, key=f"in_{who}_{k}")
                    else:
                        vals[k] = st.number_input(label, min_value=None, max_value=None, value=float(v), step=step, key=f"in_{who}_{k}")
            save = st.form_submit_button("Validate & save", type="primary")
        if save:
            errs = validate_inputs(vals)
            if not re.fullmatch(r"P-\d{2,3}", new_id.strip()):
                errs.append("Player ID must look like P-31")
            if errs:
                st.error("Not saved. Please fix:\n\n- " + "\n- ".join(errs))
                audit.log_event("entry_rejected", new_id, reviewer, {"errors": errs})
            else:
                row = {"player_id": new_id.strip(), **vals}
                sq = ss.squad[ss.squad.player_id != row["player_id"]]
                ss.squad = pd.concat([sq, pd.DataFrame([row])], ignore_index=True)
                ss.signoffs.pop(row["player_id"], None)
                audit.log_event("entry_saved", row["player_id"], reviewer, row)
                st.success(f"Saved {row['player_id']}. Risk: {score(MODEL, row, BAND)['risk_level']}.")
    else:
        cols_needed = ["player_id"] + [k for k, *_ in INPUT_FIELDS]
        st.download_button("📄 Download CSV template", ss.squad[cols_needed].head(3).to_csv(index=False),
                           "weekly_template.csv", "text/csv")
        up = st.file_uploader("Upload this week's squad CSV", type=["csv"])
        if up is not None:
            try:
                new = pd.read_csv(up)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the CSV: {str(e)[:120]}")
                new = None
            if new is not None:
                miss = [c for c in cols_needed if c not in new.columns]
                if miss:
                    st.error("Missing columns: " + ", ".join(miss))
                else:
                    good, bad = [], []
                    for r in new[cols_needed].to_dict("records"):
                        errs = validate_inputs(r)
                        (bad if errs else good).append((r, errs))
                    st.write(f"**{len(good)}** valid rows, **{len(bad)}** rejected.")
                    if bad:
                        st.dataframe(pd.DataFrame([{"player_id": r.get("player_id"), "problems": "; ".join(e)} for r, e in bad]),
                                     hide_index=True, use_container_width=True)
                    if good and st.button("Load valid rows into the squad", type="primary"):
                        gdf = pd.DataFrame([r for r, _ in good])
                        ss.squad = pd.concat([ss.squad[~ss.squad.player_id.isin(gdf.player_id)], gdf], ignore_index=True)
                        for p in gdf.player_id:
                            ss.signoffs.pop(p, None)
                        audit.log_event("csv_upload", "", reviewer, {"rows_loaded": len(gdf), "rows_rejected": len(bad)})
                        st.rerun()

# ------------------------------------------------------------------ 4. coach view preview
with tabs[3]:
    st.subheader("What the head coach sees")
    st.caption("Preview. The head coach role sees only this page: signed-off status and note, no scores or wellness data.")
    rows = []
    for rec, _ in scored:
        p = rec["player_id"]
        s = ss.signoffs.get(p)
        if s and s["data_hash"] == rec_hash(rec):
            rows.append({"Player": p, "Status": s["final_status"], "Medical note": s["coach_note"], "Signed off by": s["by"]})
        else:
            rows.append({"Player": p, "Status": "Awaiting medical sign-off", "Medical note": "—", "Signed off by": "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=500)

# ------------------------------------------------------------------ 5. evaluation
with tabs[4]:
    m = load_json(METRICS_PATH)
    st.subheader("Predictive model")
    if m:
        d = m["dataset"]
        st.markdown(f"**Data:** {d['name']} — {d['licence']} licence. {d['athletes']} real athletes, "
                    f"{d['rows']:,} athlete-days, {d['injuries']} injuries. Row = {d['row_definition'].lower()}; "
                    f"label = {d['label'].lower()}.  \n"
                    f"**Test:** {d['test_athletes']} athletes the model never saw ({d['test_rows']:,} rows, {d['test_injuries']} injuries). "
                    f"5-fold cross-validation by athlete: logistic regression AUC "
                    f"{m['cv_5fold_by_athlete']['logistic_regression']['mean_auc']} ± {m['cv_5fold_by_athlete']['logistic_regression']['sd']}.")
        tr = pd.DataFrame([{"Model": k, "AUC": v["auc"], "Flagged": f"{v['flag_rate']:.0%}", "Recall (injuries caught)": f"{v['recall']:.0%}",
                            "Precision": f"{v['precision']:.1%}", "Lift vs base rate": v["lift_vs_base_rate"],
                            "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]} for k, v in m["test_results"].items()])
        st.dataframe(tr, hide_index=True, use_container_width=True)
        st.caption(f"Base rate: {m['base_rate']:.2%} of athlete-days are followed by an injury. Precision is per athlete-day, "
                   "so it looks small; the squad projection below converts it to weekly numbers.")

    st.markdown("#### What each error costs: move the cut-off")
    tp = pd.read_csv(TEST_PRED_PATH) if os.path.exists(TEST_PRED_PATH) else None
    if tp is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            squad_n = st.number_input("Squad size", 15, 40, 30)
            weeks = st.number_input("Weeks in season", 20, 52, 42)
            inj_pp = st.number_input("Injuries per player per season", 0.5, 4.0, 2.0, 0.1,
                                     help="UEFA Elite Club Injury Study: about 2 per player per season (Ekstrand, Aspetar Journal)")
        with c2:
            days_lost = st.number_input("Days lost per injury", 3, 60, 16,
                                        help="Average return time for muscle injuries, the most common type (UEFA ECIS)")
            wage_season = st.number_input("Average wage per player per season (₹ lakh)", 5.0, 500.0, 47.0, 1.0,
                                          help="ISL squad salary cap ₹16.5 crore ÷ 35-player squad (2021-22 rules) ≈ ₹47 lakh")
        with c3:
            prevent = st.slider("Share of caught injuries prevented by the medical plan", 0.0, 1.0, 0.3, 0.05,
                                help="Team assumption, not a published figure. The biggest uncertainty in the business case.")
            fa_cost = st.number_input("Cost of one review / false alarm (₹)", 0, 50000, 5000, 500,
                                      help="Team assumption: medical staff time plus one modified session")
        thr = st.slider("Model cut-off (score)", float(np.quantile(tp.score, 0.5)), float(np.quantile(tp.score, 0.99)),
                        float(m["threshold"]) if m else float(tp.score.quantile(0.84)), 0.0005, format="%.4f")
        flag = tp.score >= thr
        recall = flag[tp.injury == 1].mean()
        flag_rate = flag.mean()
        inj_season = inj_pp * squad_n
        flags_week = flag_rate * squad_n
        caught = recall * inj_season
        missed = inj_season - caught
        day_value = wage_season * 1e5 / 365
        saved = caught * prevent * days_lost * day_value
        review = flags_week * weeks * fa_cost
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            card("Players flagged / week", f"{flags_week:.1f}", f"{flag_rate:.0%} of player-weeks")
        with k2:
            card("Injuries caught / season", f"{caught:.0f} of {inj_season:.0f}", f"recall {recall:.0%}", "#1b5e20")
        with k3:
            card("Injuries missed / season", f"{missed:.0f}", "false negatives", "#b71c1c")
        with k4:
            card("Weekly precision", f"{(caught / weeks) / flags_week:.0%}" if flags_week else "—", "flags that are real injuries")
        with k5:
            card("Net value / season", f"₹{(saved - review) / 1e5:.1f} lakh", f"saved ₹{saved / 1e5:.1f}L − reviews ₹{review / 1e5:.1f}L",
                 "#1b5e20" if saved >= review else "#b71c1c")
        st.caption("A missed injury costs about "
                   f"₹{days_lost * day_value / 1e5:.1f} lakh in paid-but-unavailable wages, before match results; a false alarm costs "
                   f"₹{fa_cost:,}. That is why the cut-off favours catching injuries over avoiding false alarms. "
                   "Recall and flag rate come from the held-out test athletes; the other numbers are the assumptions above.")

    st.subheader("LLM briefing & guardrails")
    ev = load_json(EVAL_PATH)
    if ev:
        a, b_, c = ev["A_retrieval"], ev["B_input_screen"], ev["C_output_checks"]
        x1, x2, x3 = st.columns(3)
        with x1:
            card("Retrieval", f"{a['covered_correct'] + a['uncovered_correct']}/{a['covered_total'] + a['uncovered_total']}",
                 f"{a['covered_correct']}/{a['covered_total']} covered · {a['uncovered_correct']}/{a['uncovered_total']} 'not covered'")
        with x2:
            card("Attacks blocked at input", f"{b_['blocked_v2']}/{b_['attacks_total']}",
                 f"v1 rules: {b_['blocked_v1']}/20 · normal notes wrongly blocked: {b_['benign_wrongly_blocked']}/10")
        with x3:
            card("Bad outputs caught", f"{c['correct']}/{c['cases']}", "Python output checks")
        if "D_llm_end_to_end" in ev:
            d = ev["D_llm_end_to_end"]
            y1, y2, y3 = st.columns(3)
            with y1:
                card("LLM first answer passed", f"{d['questions_first_attempt_passed']}/{d['questions']}",
                     f"re-prompted {d['questions_needed_retry']} · fallback {d['questions_fallback']}")
            with y2:
                card("'Not covered' handled", f"{d['not_covered_answered_correctly']}/{d['not_covered_total']}", d["model"])
            with y3:
                card("Attacks reaching the output", f"{d['attacks_reaching_final_output']}/{d['attacks']}",
                     f"LLM first answer broke a rule: {d['attacks_first_attempt_broke_a_rule']}/20")
            with st.expander("LLM test rows"):
                st.dataframe(pd.DataFrame(d["rows"]), hide_index=True, use_container_width=True)
        else:
            st.info("End-to-end LLM results not run yet: run `python eval/run_evaluation.py` with your Groq key.")
        with st.expander("Retrieval test rows"):
            st.dataframe(pd.DataFrame(a["rows"]), hide_index=True, use_container_width=True)
        with st.expander("Red-team attacks (input screen)"):
            st.dataframe(pd.DataFrame(b_["rows"]), hide_index=True, use_container_width=True)
        with st.expander("Output-check test cases"):
            st.dataframe(pd.DataFrame(c["rows"]), hide_index=True, use_container_width=True)
        st.caption(a["note"] + " " + b_["note"])

    st.subheader("Known limitations")
    st.markdown("""
- **Population shift:** trained on 74 elite runners, used on footballers. Running distance maps to GPS distance, but football adds contact injuries and match load the model has never seen.
- **Modest accuracy:** test AUC ≈ 0.70; most signal comes from *previous injury*, which alone scores 0.67. The model is a triage aid, not a predictor of individual injuries.
- **Missing data in the source:** 19 of 74 athletes never logged 'training success', so that feature was dropped rather than letting blanks look like low scores.
- **Sleep is not in the model** (no real data); it only triggers protocol clause C-8. The demo squad's sleep values are illustrative.
- **Cut-off drift:** set on training athletes to catch 50% of injuries; on unseen athletes it caught 31% while flagging 16%.
- **Guardrails were tuned on our own test sets**; fresh attacks (e.g. in the peer challenge) may get further.
- **Privacy:** prototype uses only public, anonymised data. Groq's free tier must not receive real player health data; a live version needs a paid tier with a data-processing agreement or a self-hosted model (DPDP Act 2023).
""")

# ------------------------------------------------------------------ 6. audit log
with tabs[5]:
    st.subheader("Audit log")
    ev_rows = audit.read_events()
    if ev_rows:
        adf = pd.DataFrame(ev_rows)
        st.dataframe(adf, hide_index=True, use_container_width=True, height=480)
        st.download_button("📥 Download audit log (CSV)", adf.to_csv(index=False), "audit_log.csv", "text/csv")
    else:
        st.info("No events yet. Generate a briefing or sign off a player.")

# ------------------------------------------------------------------ 7. protocol
with tabs[6]:
    st.subheader("Club protocol (knowledge base for retrieval)")
    st.caption("Read fresh from knowledge/protocol.md on every run. Changing a threshold in an `Applies when:` line "
               "changes which clauses are retrieved, with no code change.")
    st.dataframe(pd.DataFrame([{"Clause": c.cid, "Title": c.title, "Applies when": c.condition, "Text": c.text}
                               for c in PROTOCOL.clauses]), hide_index=True, use_container_width=True, height=420)
    with st.expander("Raw protocol.md"):
        st.code(PROTOCOL.raw, language="markdown")
