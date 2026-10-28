"""
AI-Assisted Player Health & Injury Risk System - Team 8, Project 4 (Healthcare)
Educational prototype. Not medical advice.

Run locally:  streamlit run app.py
"""

import hashlib
import hmac
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

import audit
from audit import now_ist
from briefing import available_models, choose_models, generate_briefing
from guardrails import PROMPT_GUARD_MODEL
from features import INPUT_FIELDS
from guardrails import screen_text
from protocol import NOT_COVERED, load_protocol, retrieve
from risk import facts_for_protocol, load_model, score, validate_inputs, what_if
from secrets_util import get_groq_client, get_secret

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
ROLE_ICONS = {"Sports scientist": "🧪", "Club doctor": "🩺", "Head physiotherapist": "💪", "Head coach": "📣"}
DEMO_CASES = [  # the cases the brief asks to demo, plus two edge cases
    ("P-07", "Low risk"),
    ("P-29", "High risk"),
    ("P-03", "Edge: model vs protocol"),
    ("P-17", "Edge: borderline score"),
    ("P-30", "Edge: no recent data"),
]
DISCLAIMER = ("Educational prototype for a university course — not medical advice. The risk model was trained on "
              "real data from elite runners, not footballers, and every decision is made by the club's medical team.")

st.set_page_config(page_title="Player Health & Injury Risk", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: #f8f7f4; color: #1e293b; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.2rem; }
button[data-baseweb="tab"] p { font-size: 0.95rem; font-weight: 600; }
.hero { background: linear-gradient(135deg, #0f1f3d 0%, #1e3a68 100%); color: #fff; padding: 14px 22px;
        border-radius: 12px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 8px; }
.hero h1 { font-size: 1.35rem; margin: 0; color: #fff; letter-spacing: -0.01em; padding: 0; }
.hero p { margin: 2px 0 0 0; color: #cbd5e1; font-size: 0.85rem; }
.hero .tag { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); color: #fff;
             padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; letter-spacing: 0.06em; font-weight: 600; }
.flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 10px 0; }
.flow .step { background: #fff; border: 1px solid #e2e8f0; border-radius: 999px; padding: 3px 11px; font-size: 0.8rem; color: #334155; }
.flow .step b { color: #1e3a68; margin-right: 4px; }
.flow .arrow { color: #94a3b8; font-size: 0.8rem; }
.flow .human { border-color: #1b5e20; background: #e8f5e9; }
.disclaimer { background: #fff8e1; border: 1px solid #f3d27a; color: #6b4e00; padding: 6px 12px;
              border-radius: 8px; font-size: 0.8rem; margin-bottom: 12px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; height: 100%; }
.card .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600; }
.card .value { font-size: 1.55rem; font-weight: 700; color: #0f1f3d; line-height: 1.2; }
.card .sub { font-size: 0.74rem; color: #94a3b8; }
.mono { font-family: 'DM Mono', monospace; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size: 0.8rem; margin-right: 4px; }
.pill-high { background: #ffebee; color: #b71c1c; }
.pill-low { background: #e8f5e9; color: #1b5e20; }
.pill-border { background: #fff8e1; color: #b26a00; }
.pill-data { background: #f1f5f9; color: #475569; }
.pill-gen { background: #e3f2fd; color: #0d47a1; }
.pill-wait { background: #f1f5f9; color: #475569; }
.pill-done { background: #e8f5e9; color: #1b5e20; }
.chip { display: inline-block; background: #eef2ff; color: #1e3a68; border-radius: 6px; padding: 3px 9px;
        margin: 2px 6px 2px 0; font-size: 0.82rem; }
.player-id { font-family: 'DM Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #0f1f3d; margin-right: 8px; }
.next { font-size: 0.85rem; color: #334155; margin-top: 6px; }
.clause { background: #fff; border-left: 4px solid #1e3a68; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 0.9rem; }
.coach-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
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


@st.cache_resource(show_spinner=False, ttl=3600)
def get_model_info():
    """Ask Groq once an hour which models this key can use (models get retired)."""
    main, small, options = choose_models(CLIENT)
    avail = available_models(CLIENT) if CLIENT is not None else []
    return main, small, options, (PROMPT_GUARD_MODEL in avail) if avail else None


MAIN_MODEL, SMALL_MODEL, MODEL_OPTIONS, PROMPT_GUARD_ON = get_model_info()


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
    return [(rec, score(MODEL, rec, BAND)) for rec in squad.to_dict("records")]


def get_rec(pid):
    return ss.squad[ss.squad.player_id == pid].iloc[0].to_dict()


def signoff_state(pid, rec):
    """('none' | 'stale' | 'done', record)"""
    s = ss.signoffs.get(pid)
    if not s:
        return "none", None
    if s["data_hash"] != rec_hash(rec):
        return "stale", s
    return "done", s


def signoff_label(pid, rec):
    state, s = signoff_state(pid, rec)
    return {"none": "⏳ Awaiting sign-off", "stale": "⏳ New data – re-sign"}.get(state) or f"✅ {s['final_status']}"


def risk_tag(res):
    if res["no_recent_data"]:
        return "⚪ Data check"
    return ("🔴 High" if res["risk_level"] == "High" else "🟢 Low") + (" · borderline" if res["borderline"] else "")


def latest_briefing(pid, rec):
    prev = [v for k, v in ss.briefings.items() if k.startswith(f"{pid}|{rec_hash(rec)}|")]
    return prev[-1] if prev else None


def jump_to(pid):
    ss["brief_player"] = pid


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚽ Medical Room")
    st.caption("Team 8 · Project 4 (Healthcare) · ISL club prototype")
    role = st.selectbox("I am the...", ROLES, index=0, format_func=lambda r: f"{ROLE_ICONS[r]}  {r}",
                        help="Views change with the role. Only the club doctor or head physio can sign off; "
                             "the head coach only sees signed-off outcomes (protocol C-16).")
    reviewer = st.text_input("Your name (for the audit log)", value="Demo user")
    # Access control: medical views need the staff PIN; signing off needs the sign-off PIN.
    STAFF_PIN, SIGNOFF_PIN = get_secret("staff_pin"), get_secret("signoff_pin")
    staff_ok = True
    if role != "Head coach":
        if STAFF_PIN:
            pin = st.text_input("Staff PIN", type="password",
                                help="Medical and sports-science views are PIN-protected. The coach view needs no PIN.")
            staff_ok = bool(pin) and hmac.compare_digest(pin, STAFF_PIN)
            if pin and not staff_ok:
                st.error("Wrong staff PIN.")
        else:
            st.caption("⚠️ No staff PIN set in Secrets, so anyone can open the medical views.")

    if role != "Head coach" and staff_ok:
        st.divider()
        with st.expander("🎬 Demo players", expanded=True):
            st.caption("Click a player, then open the **🩺 Player** tab.")
            known = set(ss.squad["player_id"])
            for pid_, label_ in DEMO_CASES:
                if pid_ in known:
                    st.button(f"{pid_} · {label_}", key=f"demo_{pid_}", on_click=jump_to, args=(pid_,),
                              use_container_width=True)

    st.divider()
    if CLIENT is None:
        st.warning("🤖 No Groq key found: briefings use the fixed fallback template.")
    else:
        st.success("🤖 AI briefings on (Groq key loaded)")
    llm_model = st.selectbox("Briefing model", MODEL_OPTIONS,
                             index=MODEL_OPTIONS.index(MAIN_MODEL) if MAIN_MODEL in MODEL_OPTIONS else 0,
                             help="Only models your Groq key can use right now are listed.")
    if CLIENT is not None and PROMPT_GUARD_ON is not None:
        st.caption("🛡️ Prompt Guard 2 classifier: " + ("on" if PROMPT_GUARD_ON else
                   "not available on this key (keyword screen only)"))
    if st.button("🔄 Reset demo", use_container_width=True, help="Clears squad edits, briefings and sign-offs"):
        for k in ["squad", "briefings", "signoffs", "notes", "brief_player"]:
            ss.pop(k, None)
        audit.log_event("demo_reset", actor=reviewer)
        st.rerun()
    st.caption(f"Model cut-off {MODEL['threshold']:.4f} · borderline band ±{BAND:.0%}")

# ------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><div><h1>⚽ AI-Assisted Player Health & Injury Risk</h1>'
    '<p>Decision support for the club medical team</p></div>'
    '<span class="tag">EDUCATIONAL PROTOTYPE</span></div>'
    '<div class="flow">'
    '<span class="step"><b>1</b>Weekly data</span><span class="arrow">→</span>'
    '<span class="step"><b>2</b>Risk model</span><span class="arrow">→</span>'
    '<span class="step"><b>3</b>Protocol briefing (AI)</span><span class="arrow">→</span>'
    '<span class="step human"><b>4</b>Medical sign-off (human)</span><span class="arrow">→</span>'
    '<span class="step"><b>5</b>Coach view</span></div>',
    unsafe_allow_html=True)
st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

scored = score_squad(ss.squad)
player_ids = list(ss.squad["player_id"])


def color_status(v):
    v = str(v)
    if v.startswith("Train as planned"):
        return "background-color:#e8f5e9;color:#1b5e20;font-weight:600"
    if v.startswith("Modified load"):
        return "background-color:#fff8e1;color:#b26a00;font-weight:600"
    if v.startswith("Medical assessment"):
        return "background-color:#ffebee;color:#b71c1c;font-weight:600"
    return "color:#64748b"


def render_coach_table(height=560, show_signer=True):
    rows = []
    for rec, _ in scored:
        p = rec["player_id"]
        state, s = signoff_state(p, rec)
        if state == "done":
            row = {"Player": p, "Status": s["final_status"], "Medical note": s["coach_note"]}
            if show_signer:
                row.update({"Signed off by": f"{s['by']} ({s['role']})", "Signed at": s["at"]})
        else:
            row = {"Player": p, "Status": "Awaiting medical sign-off", "Medical note": "—"}
            if show_signer:
                row.update({"Signed off by": "—", "Signed at": "—"})
        rows.append(row)
    cdf = pd.DataFrame(rows)
    done = cdf[cdf.Status != "Awaiting medical sign-off"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Signed off", len(done), f"of {len(cdf)} players", "#1b5e20")
    with c2:
        card("Train as planned", int((done.Status == "Train as planned").sum()), "no change", "#1b5e20")
    with c3:
        card("Modified load", int(done.Status.str.startswith("Modified").sum()), "medical plan", "#b26a00")
    with c4:
        card("Assessment first", int(done.Status.str.startswith("Medical assessment").sum()),
             "before next session", "#b71c1c")
    st.write("")
    st.dataframe(cdf.style.map(color_status, subset=["Status"]), use_container_width=True, hide_index=True,
                 height=height, column_config={"Medical note": st.column_config.TextColumn(width="large")})


# ================================================================== HEAD COACH VIEW
if role == "Head coach" or not staff_ok:
    if not staff_ok:
        st.warning("🔒 Enter the staff PIN in the sidebar to open the medical views. Until then you see the coach view.")
    st.subheader("📣 Coach view: this week's signed-off status")
    st.caption("Only outcomes signed off by the medical team appear here. Risk scores and wellness data are not "
               "shown to coaching staff (protocol C-16).")
    render_coach_table(height=620)
    st.stop()

# ================================================================== STAFF VIEWS
tabs = st.tabs(["📋 Squad", "🩺 Player", "✍️ Data entry", "👀 Coach view", "📊 Evaluation", "🗂️ Audit log", "📜 Protocol"])

# ------------------------------------------------------------------ 1. squad dashboard
with tabs[0]:
    n_high = sum(r["risk_level"] == "High" and not r["no_recent_data"] for _, r in scored)
    n_border = sum(r["borderline"] and not r["no_recent_data"] for _, r in scored)
    n_data = sum(r["no_recent_data"] for _, r in scored)
    n_signed = sum(signoff_state(rec["player_id"], rec)[0] == "done" for rec, _ in scored)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("Players", len(scored), "this week's entries")
    with c2:
        card("High risk", n_high, "model flag", "#b71c1c")
    with c3:
        card("Borderline", n_border, f"±{BAND:.0%} of cut-off" + (f" · {n_data} data check" if n_data else ""), "#b26a00")
    with c4:
        card("To sign off", len(scored) - n_signed, "medical team")
    with c5:
        card("Signed off", n_signed, "visible to coach", "#1b5e20")
    st.write("")

    rows = []
    for rec, res in scored:
        facts = facts_for_protocol(rec, res)
        _, info = retrieve(PROTOCOL, dict(facts))
        rows.append({
            "Player": rec["player_id"],
            "Risk": risk_tag(res),
            "Risk index": round(res["risk_index"], 1),
            "Sign-off": signoff_label(rec["player_id"], rec),
            "Main risk factor": res["reasons"][0] if res["reasons"] else "—",
            "ACWR": None if res["no_recent_data"] else round(res["features"]["acwr"], 2),
            "Distance (km)": rec["km_this_week"],
            "Sleep (h)": rec["sleep_hours"],
            "Protocol flags": ", ".join(c for c in info["rule_clauses"] if c not in ("C-1", "C-2", "C-3", "C-16")) or "—",
            "Conflict": "⚠️ " + ", ".join(f"{a}/{b}" for a, b in info["conflict_pairs"]) if info["conflict_pairs"] else "",
        })
    df = pd.DataFrame(rows).sort_values("Risk index", ascending=False)

    f1, f2 = st.columns([2, 3])
    with f1:
        view = st.segmented_control("Show", ["All", "Needs attention", "Awaiting sign-off"], default="All",
                                    key="squad_filter") or "All"
    with f2:
        search = st.text_input("Search player", placeholder="e.g. P-29", label_visibility="visible").strip()
    shown = df
    if view == "Needs attention":
        shown = df[df.Risk.str.contains("High|borderline|Data check") | (df.Conflict != "")]
    elif view == "Awaiting sign-off":
        shown = df[~df["Sign-off"].str.startswith("✅")]
    if search:
        shown = shown[shown.Player.str.contains(search, case=False)]

    def color_risk(v):
        v = str(v)
        if "Data check" in v:
            return "background-color:#f1f5f9;color:#475569;font-weight:600"
        if "High" in v:
            return "background-color:#ffebee;color:#b71c1c;font-weight:600"
        if "borderline" in v:
            return "background-color:#fff8e1;color:#b26a00;font-weight:600"
        if "Low" in v:
            return "background-color:#e8f5e9;color:#1b5e20;font-weight:600"
        return ""

    def color_signoff(v):
        return "color:#1b5e20;font-weight:600" if str(v).startswith("✅") else "color:#64748b"

    if shown.empty:
        st.info("No players match this filter.")
    else:
        styled = shown.style.map(color_risk, subset=["Risk"]).map(color_signoff, subset=["Sign-off"]).format(
            {"Risk index": "{:.1f}×", "ACWR": lambda x: "—" if pd.isna(x) else f"{x:.2f}",
             "Distance (km)": "{:.1f}", "Sleep (h)": "{:.1f}"})
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(560, 38 + 35 * len(shown)),
                     column_config={
                         "Risk index": st.column_config.Column(help="Model risk ÷ average player-week. 2.0× = twice the average."),
                         "ACWR": st.column_config.Column(help="This week's distance ÷ average of the last 3 weeks"),
                         "Main risk factor": st.column_config.TextColumn(width="medium"),
                         "Protocol flags": st.column_config.TextColumn(help="Extra protocol clauses triggered by this player's numbers"),
                     })
    st.caption(f"Showing {len(shown)} of {len(df)} players. Squad = 30 real athlete-weeks from held-out test athletes "
               "(anonymised P-01…P-30). Sleep values are illustrative: the dataset has no sleep data, so sleep only "
               "drives protocol clause C-8, not the model. To open a player, use **🎬 Demo players** in the sidebar or the **🩺 Player** tab.")
    st.download_button("📥 Download squad table (CSV)", df.to_csv(index=False), "squad_risk.csv", "text/csv")

# ------------------------------------------------------------------ 2. player page
with tabs[1]:
    order = [r["player_id"] for r, _ in sorted(scored, key=lambda t: -t[1]["probability"])]
    if ss.get("brief_player") not in order:
        ss.pop("brief_player", None)
    res_by_id = {r["player_id"]: s for r, s in scored}

    def player_label(p):
        return f"{p}  ·  {risk_tag(res_by_id[p])}  ·  {signoff_label(p, get_rec(p))}"

    pid = st.selectbox("Player (sorted by risk)", order, key="brief_player", format_func=player_label)
    rec = get_rec(pid)
    res = score(MODEL, rec, BAND)
    facts = facts_for_protocol(rec, res)
    _, rule_info = retrieve(PROTOCOL, dict(facts))
    state, sign = signoff_state(pid, rec)
    existing_briefing = latest_briefing(pid, rec)

    # ---- player header ----
    with st.container(border=True):
        h1, h2, h3, h4, h5 = st.columns([2.2, 1, 1, 1, 1])
        with h1:
            lvl = res["risk_level"]
            pills = (f'<span class="pill pill-data">Data check</span>' if res["no_recent_data"] else
                     f'<span class="pill {"pill-high" if lvl == "High" else "pill-low"}">{lvl} risk</span>')
            if res["borderline"] and not res["no_recent_data"]:
                pills += '<span class="pill pill-border">borderline</span>'
            pills += ('<span class="pill pill-done">✅ Signed off</span>' if state == "done" else
                      '<span class="pill pill-wait">⏳ Awaiting sign-off</span>')
            if state == "done":
                nxt = f"Done: <b>{sign['final_status']}</b> is visible to the coach."
            elif existing_briefing is None:
                nxt = "Next step: generate the protocol briefing (📄 Briefing tab below)."
            else:
                nxt = "Next step: medical sign-off (✅ Sign-off tab below)."
            st.markdown(f'<span class="player-id">{pid}</span>{pills}<div class="next">{nxt}</div>',
                        unsafe_allow_html=True)
        with h2:
            card("Risk index", f"{res['risk_index']:.1f}×", "vs average week",
                 "#b71c1c" if lvl == "High" else "#0f1f3d")
        with h3:
            pct = res["percentile"]
            card("Riskier than", f"{min(pct, 99)}%", "of training weeks")
        with h4:
            card("ACWR", "—" if res["no_recent_data"] else f"{res['features']['acwr']:.2f}", "this week ÷ 3-wk avg")
        with h5:
            card("Sleep", f"{rec['sleep_hours']:g} h", "per night (protocol only)",
                 "#b71c1c" if rec["sleep_hours"] < 7 else "#0f1f3d")
        reasons = res["reasons"] or ["no factor pushes the risk above average"]
        st.markdown("**Why the model rates him this way:** " + "".join(f'<span class="chip">{r}</span>' for r in reasons),
                    unsafe_allow_html=True)
        if res["no_recent_data"]:
            st.warning("No training distance logged in the last 3 weeks, so the workload ratio can't be calculated. "
                       "Check the data before relying on this score (protocol C-15).")
        if rule_info["conflict_pairs"]:
            pairs = ", ".join(f"{a} vs {b}" for a, b in rule_info["conflict_pairs"])
            st.warning(f"⚠️ Protocol clauses point in different directions ({pairs}). Clause C-17 applies: "
                       "no load increase and a medical review within 24 hours until the club doctor decides.")

    sub = st.tabs(["📄 Briefing", "✅ Sign-off", "🔮 What-if", "🔢 This week's data"])

    # ---- briefing ----
    with sub[0]:
        n1, n2 = st.columns(2)
        with n1:
            notes = st.text_area("Staff notes (optional, treated as data, never as instructions)",
                                 value=ss.notes.get(pid, ""), max_chars=600, key=f"notes_{pid}", height=96)
        with n2:
            question = st.text_area("Question for the protocol (optional)", max_chars=300, key=f"q_{pid}", height=96,
                                    placeholder="e.g. How much can we reduce his load next week?")
        go = st.button("✨ Generate briefing", type="primary")
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
                             f"({why[0][:90]}). Nothing was sent to the AI.")
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
                out["created"] = now_ist()
                ss.briefings[key] = out
                audit.log_event("briefing_generated", pid, reviewer, {
                    "generated_by": out["generated_by"], "attempts": out["attempts"],
                    "guardrail_problems": out["problems"], "clauses": [c[0] for c in out["clauses"]],
                    "risk_level": res["risk_level"], "risk_index": round(res["risk_index"], 2),
                    "question": question, "notes": notes})

        out = ss.briefings.get(key) or latest_briefing(pid, rec)
        if out:
            b = out["briefing"]
            with st.container(border=True):
                st.markdown(f'<span class="pill pill-gen">{out["generated_by"]}</span>'
                            f'<span style="color:#64748b;font-size:0.8rem">{out["attempts"]} attempt(s) · {out["created"]}'
                            f'{" · " + out["model_used"] if out.get("model_used") else ""}</span>',
                            unsafe_allow_html=True)
                st.markdown(f"**Summary.** {b['summary']}")
                if b.get("key_factors"):
                    st.markdown("".join(f'<span class="chip">{k}</span>' for k in b["key_factors"]),
                                unsafe_allow_html=True)
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
            with st.expander("🔍 Retrieved clauses & guardrail log"):
                st.markdown(f"Matched by the player's numbers: **{', '.join(out['info']['rule_clauses'])}**  \n"
                            f"Matched by the question: **{', '.join(out['info']['question_clauses']) or '—'}** "
                            f"(similarity {out['info']['question_similarities'] or '—'})")
                for cid, title, text in out["clauses"]:
                    st.markdown(f"- **{cid} {title}** — {text}")
                if out["problems"]:
                    st.markdown("**Guardrail checks that failed along the way** (the answer was re-prompted or replaced):")
                    for p in out["problems"]:
                        st.markdown(f"- {p}")
                else:
                    st.markdown("✅ All output checks passed on the first attempt.")
            if state != "done":
                st.caption("Next: open the **✅ Sign-off** tab above.")
        else:
            st.info("Press **✨ Generate briefing** to retrieve the relevant protocol clauses and draft a briefing "
                    "for the medical team.")

    # ---- sign-off (human checkpoint) ----
    with sub[1]:
        out = latest_briefing(pid, rec)
        if state == "done":
            st.success(f"Signed off by **{sign['by']}** ({sign['role']}) at {sign['at']}: "
                       f"**{sign['final_status']}** — decision: {sign['decision']}")
            st.markdown(f"Note for the coach: _{sign['coach_note']}_")
            if sign.get("reason"):
                st.caption(f"Reason: {sign['reason']}")
        elif state == "stale":
            st.warning("New weekly data was entered after the last sign-off, so it no longer applies. Please re-sign.")
        if role not in SIGNOFF_ROLES:
            st.info(f"🔒 Only the club doctor or head physiotherapist can sign off. You are signed in as "
                    f"**{role}**: switch role in the sidebar.")
        elif out is None:
            st.info("Generate the briefing first (📄 Briefing tab), so the sign-off is based on it.")
        else:
            st.caption(f"Model: **{res['risk_level']} risk** ({res['risk_index']:.1f}×) · briefing by "
                       f"{out['generated_by']}. The decision below is yours, not the model's.")
            with st.form(f"signoff_{pid}"):
                decision = st.radio("Your decision on the briefing", ["Approve", "Modify", "Override"], horizontal=True,
                                    help="Approve = accept as drafted. Modify = change the plan. "
                                         "Override = disagree with the model's risk level.")
                # No pre-selected status, so the reviewer makes an active choice (reduces automation bias).
                final_status = st.selectbox("Training status for this week", STATUS_OPTIONS,
                                            index=None, placeholder="Choose a status...")
                coach_note = st.text_area("Note the coach will see (no wellness scores or diagnoses, C-16)",
                                          max_chars=400,
                                          placeholder="e.g. Modified load this week; available for selection decisions on Friday.")
                reason = st.text_area("Reason (required for Modify or Override)", max_chars=400)
                entered_pin = st.text_input("Sign-off PIN (club doctor / head physio only)", type="password") \
                    if SIGNOFF_PIN else None
                submitted = st.form_submit_button("✅ Sign off", type="primary")
            if submitted:
                if SIGNOFF_PIN and not (entered_pin and hmac.compare_digest(entered_pin, SIGNOFF_PIN)):
                    st.error("Wrong or missing sign-off PIN. Nothing was signed.")
                    audit.log_event("signoff_rejected_pin", pid, f"{reviewer} ({role})")
                elif final_status is None or not coach_note.strip():
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
                                            "at": now_ist(), "data_hash": rec_hash(rec),
                                            "model_level": res["risk_level"], "briefing_by": out["generated_by"]}
                        audit.log_event("signoff", pid, f"{reviewer} ({role})", ss.signoffs[pid])
                        st.rerun()

    # ---- what-if ----
    with sub[2]:
        w1, w2 = st.columns([1.3, 1])
        with w1:
            cut = st.slider("Reduce next week's distance by (%)", 0, 50, 20, 5)
            hcut = st.slider("Reduce high-intensity distance by (%)", 0, 100, 30, 10)
            extra = st.slider("Extra rest days", 0, 2, 1)
        same = what_if(MODEL, rec, 0, 0, 0)
        plan = what_if(MODEL, rec, cut, extra, hcut)
        with w2:
            st.metric("Same load next week", f"{same['risk_index']:.1f}×", same["risk_level"], delta_color="off")
            st.metric("With this plan", f"{plan['risk_index']:.1f}×",
                      f"{plan['risk_index'] - same['risk_index']:+.1f}× · {plan['risk_level']}", delta_color="inverse")
        st.caption("The what-if uses the same model. It shows an association, not proof that reducing load prevents "
                   "injury; protocol clause C-7 warns that cutting load too far (ACWR below 0.8) also raises risk.")

    # ---- this week's data ----
    with sub[3]:
        st.dataframe(pd.DataFrame([(lbl, rec[k]) for k, lbl, *_ in INPUT_FIELDS], columns=["Field", "Value"]),
                     hide_index=True, use_container_width=True)
        st.caption("To change these numbers, use the **✍️ Data entry** tab. New data resets the player's sign-off.")

# ------------------------------------------------------------------ 3. data entry
FIELD_GROUPS = [
    ("🏃 Training load", ["km_this_week", "km_last_week", "km_two_weeks_ago", "hi_km"]),
    ("😴 Recovery & wellness", ["rest_days", "strength_sessions", "exertion", "sleep_hours"]),
    ("🩹 History", ["prior_injury_12m"]),
]
FIELD_INFO = {k: (label, lo, hi, default, step) for k, label, lo, hi, default, step in INPUT_FIELDS}

with tabs[2]:
    st.subheader("Weekly data entry")
    st.caption("Every field is range-checked before scoring (input guardrail). New data for a player resets "
               "their sign-off, so an old approval never reaches the coach.")
    mode = st.segmented_control("Entry mode", ["One player", "Whole squad (CSV)"], default="One player",
                                key="entry_mode") or "One player"
    if mode == "One player":
        existing = ["➕ New player"] + player_ids
        who = st.selectbox("Player to add or update", existing, key="entry_player")
        base = get_rec(who) if who != "➕ New player" else {}
        with st.form("entry"):
            new_id = st.text_input("Player ID", value=who if who != "➕ New player" else f"P-{len(player_ids) + 1:02d}")
            vals = {}
            for title, keys in FIELD_GROUPS:
                st.markdown(f"**{title}**")
                cols = st.columns(4)
                for i, k in enumerate(keys):
                    label, lo, hi, default, step = FIELD_INFO[k]
                    v = base.get(k, default)
                    with cols[i % 4]:
                        if k == "prior_injury_12m":
                            vals[k] = 1 if st.radio("Time-loss injury in the past 12 months?", ["No", "Yes"],
                                                    index=int(float(v)), horizontal=True, key=f"in_{who}_{k}") == "Yes" else 0
                        elif isinstance(step, int):
                            vals[k] = st.number_input(label, value=int(v), step=step, key=f"in_{who}_{k}")
                        else:
                            vals[k] = st.number_input(label, value=float(v), step=step, key=f"in_{who}_{k}")
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
                st.success(f"Saved {row['player_id']}. Model risk: {score(MODEL, row, BAND)['risk_level']}. "
                           "Open the 🩺 Player tab to brief and sign off.")
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
    st.caption("Preview. The head coach role sees only this page: signed-off status and note — no scores or wellness data.")
    render_coach_table(height=520)

# ------------------------------------------------------------------ 5. evaluation
with tabs[4]:
    m = load_json(METRICS_PATH)
    ev = load_json(EVAL_PATH)
    etabs = st.tabs(["🎯 Risk model", "💰 Cost of errors", "🤖 AI & guardrails", "⚠️ Limitations"])

    with etabs[0]:
        if m:
            d = m["dataset"]
            lr = m["test_results"][next(k for k in m["test_results"] if k.startswith("Logistic regression, chosen"))]
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                card("AUC on unseen athletes", f"{lr['auc']:.2f}",
                     f"5-fold CV {m['cv_5fold_by_athlete']['logistic_regression']['mean_auc']} ± "
                     f"{m['cv_5fold_by_athlete']['logistic_regression']['sd']}")
            with k2:
                card("Injuries caught", f"{lr['recall']:.0%}", "recall at the chosen cut-off", "#1b5e20")
            with k3:
                card("Player-weeks flagged", f"{lr['flag_rate']:.0%}", "medical reviews needed")
            with k4:
                card("Lift over base rate", f"{lr['lift_vs_base_rate']:.1f}×", "flagged weeks vs average")
            st.write("")
            st.markdown(f"**Data:** {d['name']} — {d['licence']} licence (DOI 10.34894/UWU9PV). {d['athletes']} real athletes, "
                        f"{d['rows']:,} athlete-days, {d['injuries']} injuries. Row = {d['row_definition'].lower()}; "
                        f"label = {d['label'].lower()}. **Test:** {d['test_athletes']} athletes the model never saw "
                        f"({d['test_rows']:,} rows, {d['test_injuries']} injuries).")
            tr = pd.DataFrame([{"Model": k, "AUC": v["auc"], "Flagged": f"{v['flag_rate']:.0%}",
                                "Recall (injuries caught)": f"{v['recall']:.0%}", "Precision": f"{v['precision']:.1%}",
                                "Lift vs base rate": v["lift_vs_base_rate"], "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]}
                               for k, v in m["test_results"].items()])
            st.dataframe(tr, hide_index=True, use_container_width=True)
            st.caption(f"Base rate: {m['base_rate']:.2%} of athlete-days are followed by an injury, so per-day precision "
                       "looks small; the **💰 Cost of errors** tab converts it to weekly squad numbers.")
        else:
            st.info("Model metrics not found: run `python train_model.py`.")

    with etabs[1]:
        tp = pd.read_csv(TEST_PRED_PATH) if os.path.exists(TEST_PRED_PATH) else None
        if tp is None:
            st.info("Test predictions not found: run `python train_model.py`.")
        else:
            st.markdown("Move the **cut-off** and the assumptions to see the trade-off between missed injuries and "
                        "false alarms for a 30-player squad.")
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
                card("Flagged / week", f"{flags_week:.1f}", f"{flag_rate:.0%} of player-weeks")
            with k2:
                card("Injuries caught", f"{caught:.0f} of {inj_season:.0f}", f"per season · recall {recall:.0%}", "#1b5e20")
            with k3:
                card("Injuries missed", f"{missed:.0f}", "per season (false negatives)", "#b71c1c")
            with k4:
                card("Weekly precision", f"{(caught / weeks) / flags_week:.0%}" if flags_week else "—", "flags that are real injuries")
            with k5:
                card("Net value", f"₹{(saved - review) / 1e5:.1f}L", f"saved ₹{saved / 1e5:.1f}L − reviews ₹{review / 1e5:.1f}L",
                     "#1b5e20" if saved >= review else "#b71c1c")
            st.caption("A missed injury costs about "
                       f"₹{days_lost * day_value / 1e5:.1f} lakh in paid-but-unavailable wages, before match results; a false "
                       f"alarm costs ₹{fa_cost:,}. That is why the cut-off favours catching injuries over avoiding false alarms. "
                       "Recall and flag rate come from the held-out test athletes; the other numbers are the assumptions above.")

    with etabs[2]:
        if not ev:
            st.info("Evaluation results not found: run `python eval/run_evaluation.py`.")
        else:
            a, b_, c = ev["A_retrieval"], ev["B_input_screen"], ev["C_output_checks"]
            st.markdown("**Offline tests** (no AI calls)")
            x1, x2, x3 = st.columns(3)
            with x1:
                card("Retrieval", f"{a['covered_correct'] + a['uncovered_correct']}/{a['covered_total'] + a['uncovered_total']}",
                     f"{a['covered_correct']}/{a['covered_total']} covered · {a['uncovered_correct']}/{a['uncovered_total']} 'not covered'")
            with x2:
                card("Attacks blocked at input", f"{b_['blocked_v2']}/{b_['attacks_total']}",
                     f"v1 rules {b_['blocked_v1']}/20 · normal notes wrongly blocked {b_['benign_wrongly_blocked']}/10")
            with x3:
                card("Planted bad outputs caught", f"{c['correct']}/{c['cases']}",
                     f"Python output checks (v1 checks: {c.get('correct_v1', c['correct'])}/{c['cases']})")
            if "D_llm_end_to_end" in ev:
                d = ev["D_llm_end_to_end"]
                st.markdown(f"**Live AI tests** ({d['model']})")
                y1, y2, y3 = st.columns(3)
                with y1:
                    card("First answer passed", f"{d['questions_first_attempt_passed']}/{d['questions']}",
                         f"re-prompted {d['questions_needed_retry']} · fallback {d['questions_fallback']}")
                with y2:
                    card("'Not covered' handled", f"{d['not_covered_answered_correctly']}/{d['not_covered_total']}",
                         "off-topic questions")
                with y3:
                    card("Attacks reaching the output", f"{d['attacks_reaching_final_output']}/{d['attacks']}",
                         f"AI's first answer broke a rule: {d['attacks_first_attempt_broke_a_rule']}/20")
            else:
                st.info("Live AI results not run yet: run `python eval/run_evaluation.py` with your Groq key.")
            if "E_hard_attacks" in ev:
                e = ev["E_hard_attacks"]
                st.markdown(f"**Harder attacks** ({e['attacks']} subtle attacks on {e['model']}: Hinglish, contradicting the "
                            "risk level, misusing clause numbers, treatment advice, comparisons with other players)")
                z1, z2, z3 = st.columns(3)
                with z1:
                    card("Blocked at input", f"{e['blocked_at_input']}/{e['attacks']}", "keyword screen")
                with z2:
                    card("AI answers that broke a rule", f"{e['llm_answers_flagged_v2']}/{e['attacks']}",
                         f"old checks caught {e['llm_answers_flagged_v1']}", "#b71c1c")
                with z3:
                    card("Passed every check", f"{e['passed_all_checks_v2']}/{e['attacks']}",
                         "read by hand: did any obey the attack?")
                st.caption(e["note"])
            if "E_hard_attacks_small" in ev:
                es = ev["E_hard_attacks_small"]
                st.markdown(f"**Same harder attacks on the small model** ({es['model']})")
                s1, s2, s3 = st.columns(3)
                with s1:
                    card("Blocked at input", f"{es['blocked_at_input']}/{es['attacks']}", "keyword screen")
                with s2:
                    card("AI answers that broke a rule", f"{es['llm_answers_flagged_v2']}/{es['attacks']}",
                         f"old checks caught {es['llm_answers_flagged_v1']}", "#b71c1c")
                with s3:
                    card("Passed every check", f"{es['passed_all_checks_v2']}/{es['attacks']}",
                         "read by hand: did any obey the attack?")
            st.markdown("**Test rows**")
            if "D_llm_end_to_end" in ev:
                with st.expander("Live AI test rows"):
                    st.dataframe(pd.DataFrame(ev["D_llm_end_to_end"]["rows"]), hide_index=True, use_container_width=True)
            if "E_hard_attacks" in ev:
                with st.expander("Harder attack rows (AI's first answer)"):
                    st.dataframe(pd.DataFrame(ev["E_hard_attacks"]["rows"]), hide_index=True, use_container_width=True)
            if "E_hard_attacks_small" in ev:
                with st.expander("Harder attack rows: small model"):
                    st.dataframe(pd.DataFrame(ev["E_hard_attacks_small"]["rows"]), hide_index=True, use_container_width=True)
            with st.expander("Retrieval test rows"):
                st.dataframe(pd.DataFrame(a["rows"]), hide_index=True, use_container_width=True)
            with st.expander("Red-team attacks (input screen)"):
                st.dataframe(pd.DataFrame(b_["rows"]), hide_index=True, use_container_width=True)
            with st.expander("Output-check test cases"):
                st.dataframe(pd.DataFrame(c["rows"]), hide_index=True, use_container_width=True)
            st.caption(a["note"] + " " + b_["note"])

    with etabs[3]:
        st.markdown("""
- **Population shift:** trained on 74 elite runners, used on footballers. Running distance maps to GPS distance, but football adds contact injuries and match load the model has never seen.
- **Modest accuracy:** test AUC ≈ 0.70; most signal comes from *previous injury*, which alone scores 0.67. The model is a triage aid, not a predictor of individual injuries.
- **Missing data in the source:** 19 of 74 athletes never logged 'training success', so that feature was dropped rather than letting blanks look like low scores.
- **Sleep is not in the model** (no real data); it only triggers protocol clause C-8. The demo squad's sleep values are illustrative.
- **Cut-off drift:** set on training athletes to catch 50% of injuries; on unseen athletes it caught 31% while flagging 16%.
- **Guardrails were tuned on our own test sets**; fresh attacks (e.g. in the peer challenge) may get further. Checks catch broken rules, not subtle changes of meaning — a human still reads every briefing before sign-off.
- **Access control is a prototype:** one shared staff PIN and one sign-off PIN, not individual accounts. A live system needs personal logins.
- **Sign-offs live in memory:** they reset when the app restarts; the audit log file also resets on Streamlit Cloud. A live version needs a database.
- **Privacy:** prototype uses only public, anonymised data. Groq's free tier must not receive real player health data; a live version needs a paid tier with a data-processing agreement or a self-hosted model (DPDP Act 2023).
""")

# ------------------------------------------------------------------ 6. audit log
EVENT_NAMES = {
    "briefing_generated": "📄 Briefing generated", "signoff": "✅ Signed off", "input_blocked": "🛡️ Input blocked",
    "signoff_rejected_pin": "🔒 Wrong sign-off PIN", "entry_saved": "✍️ Data saved", "entry_rejected": "⛔ Data rejected",
    "csv_upload": "📤 CSV uploaded", "demo_reset": "🔄 Demo reset",
}


def summarise_event(event, details):
    try:
        d = json.loads(details or "{}")
    except ValueError:
        return ""
    if event == "briefing_generated":
        s = f"{d.get('generated_by')} · {d.get('attempts')} attempt(s) · clauses {', '.join(d.get('clauses', []))}"
        return s + (f" · {len(d['guardrail_problems'])} check(s) failed on the way" if d.get("guardrail_problems") else "")
    if event == "signoff":
        return f"{d.get('decision')} → {d.get('final_status')} (model said {d.get('model_level')})"
    if event == "input_blocked":
        return "; ".join(f"{lbl}: {why[0][:60]}" for lbl, why in d.get("blocked", []))
    if event == "entry_rejected":
        return f"{len(d.get('errors', []))} problem(s): " + "; ".join(d.get("errors", []))[:120]
    if event == "entry_saved":
        return f"{d.get('km_this_week')} km this week · exertion {d.get('exertion')}/10 · sleep {d.get('sleep_hours')} h"
    if event == "csv_upload":
        return f"{d.get('rows_loaded')} rows loaded, {d.get('rows_rejected')} rejected"
    return ""


with tabs[5]:
    st.subheader("Audit log")
    st.caption("Every briefing, blocked input, data change and sign-off, newest first (times in IST).")
    ev_rows = audit.read_events()
    if ev_rows:
        adf = pd.DataFrame(ev_rows)
        adf["What happened"] = [summarise_event(e, d) for e, d in zip(adf.Event, adf.Details)]
        adf["Event"] = adf.Event.map(lambda e: EVENT_NAMES.get(e, e))
        pick = st.multiselect("Filter events", sorted(adf.Event.unique()), placeholder="All events")
        view_df = adf[adf.Event.isin(pick)] if pick else adf
        st.dataframe(view_df[["Timestamp", "Event", "Player", "By", "What happened"]], hide_index=True,
                     use_container_width=True, height=460,
                     column_config={"What happened": st.column_config.TextColumn(width="large")})
        st.download_button("📥 Download full audit log (CSV, with details)", adf.to_csv(index=False),
                           "audit_log.csv", "text/csv")
    else:
        st.info("No events yet. Generate a briefing or sign off a player.")

# ------------------------------------------------------------------ 7. protocol
with tabs[6]:
    st.subheader("Club protocol (knowledge base for retrieval)")
    st.caption("Read fresh from knowledge/protocol.md on every run. Changing a threshold in an `Applies when:` line "
               "changes which clauses are retrieved, with no code change.")
    st.dataframe(pd.DataFrame([{"Clause": c.cid, "Title": c.title, "Applies when": c.condition, "Text": c.text}
                               for c in PROTOCOL.clauses]), hide_index=True, use_container_width=True, height=420,
                 column_config={"Text": st.column_config.TextColumn(width="large")})
    with st.expander("Raw protocol.md"):
        st.code(PROTOCOL.raw, language="markdown")
