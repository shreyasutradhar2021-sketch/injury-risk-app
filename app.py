"""
AI-Assisted Player Health & Injury Risk System - Team 8, Project 4 (Healthcare)
Educational prototype. Not medical advice.

Run locally:  streamlit run app.py
"""

import hashlib
import hmac
import html
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import audit
from acceptance import acceptance_suite, failure_mode_tests
from audit import now_ist
from briefing import available_models, choose_models, generate_briefing
from guardrails import PROMPT_GUARD_MODEL
from features import INPUT_FIELDS
from guardrails import screen_coach_note, screen_text
from protocol import NOT_COVERED, load_protocol, retrieve
from risk import facts_for_protocol, load_model, score, validate_inputs, what_if
from secrets_util import get_groq_client, get_secret
from workflow import COACH_MESSAGES, SIGNOFF_ROLES, STATUS_OPTIONS, decision_state, signoff_problem

ROOT = os.path.dirname(os.path.abspath(__file__))
SQUAD_PATH = os.path.join(ROOT, "data", "demo_squad.csv")
METRICS_PATH = os.path.join(ROOT, "eval", "model_metrics.json")
EVAL_PATH = os.path.join(ROOT, "eval", "evaluation_results.json")
TEST_PRED_PATH = os.path.join(ROOT, "eval", "test_predictions.csv")

MAX_PIN_TRIES = 5   # sign-off rules, statuses and coach messages live in workflow.py (tested directly)
DAYS_VS_WEEKS = ("**How a score becomes this week's flag:** each row in the data is one athlete on one day, with the "
                 "3 weeks of training before it, labelled by whether a time-loss injury happened the next day. Staff "
                 "enter one weekly record per player, which is scored the same way, so the score is the injury risk "
                 "on the day right after that 3-week window; the app uses it as this week's flag. Weekly squad numbers "
                 "assume one check per player per week.")
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
                   initial_sidebar_state="auto")   # open on laptops, closed on phones

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
.pill-done { background: #e0e7ff; color: #1e3a68; }   /* a recorded decision, not 'safe' */
.chip { display: inline-block; background: #eef2ff; color: #1e3a68; border-radius: 6px; padding: 3px 9px;
        margin: 2px 6px 2px 0; font-size: 0.82rem; }
.player-id { font-family: 'DM Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #0f1f3d; margin-right: 8px; }
.next { font-size: 0.85rem; color: #334155; margin-top: 6px; }
.clause { background: #fff; border-left: 4px solid #1e3a68; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 0.9rem; }
.clause details { margin-top: 4px; } .clause summary { color: #64748b; font-size: 0.78rem; cursor: pointer; }
.clause .src { color: #334155; font-size: 0.84rem; background: #f8fafc; border-radius: 4px; padding: 6px 8px; margin-top: 4px; }
.draft { display: inline-block; background: #fff8e1; color: #6b4e00; border: 1px solid #f3d27a; border-radius: 6px;
         padding: 2px 10px; font-weight: 600; font-size: 0.8rem; margin-right: 6px; }
.vstat { font-size: 0.82rem; color: #334155; margin: 4px 0 8px 0; }
@media (max-width: 640px) {
  .hero h1 { font-size: 1.05rem; } .hero p { font-size: 0.78rem; } .flow .step { font-size: 0.72rem; padding: 2px 8px; }
  .card .value { font-size: 1.25rem; } .player-id { font-size: 1.3rem; }
}
.page-h { font-size: 1.6rem; font-weight: 700; color: #0f1f3d; margin: 4px 0 0 0; letter-spacing: -0.01em; }
.page-sub { font-size: 0.85rem; color: #64748b; margin: 0 0 10px 0; }
.sec-h { font-size: 1.15rem; font-weight: 600; color: #0f1f3d; margin: 16px 0 6px 0; }
.queue { background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #94a3b8; border-radius: 10px;
         padding: 8px 12px; margin-bottom: 4px; min-height: 62px; }
.queue.q-high { border-left-color: #b71c1c; } .queue.q-conflict { border-left-color: #b26a00; }
.queue.q-data { border-left-color: #64748b; } .queue.q-border { border-left-color: #d97706; }
.queue .qid { font-family: 'DM Mono', monospace; font-weight: 600; font-size: 1.05rem; color: #0f1f3d; }
.queue .qwhy { font-size: 0.8rem; color: #475569; }
.stepper { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px 0; }
.stepper .s { flex: 1 1 140px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 6px 10px; background: #fff;
              font-size: 0.76rem; color: #64748b; }
.stepper .s b { display: block; font-size: 0.86rem; color: #334155; }
.stepper .s.done { background: #f1f5f9; }
.stepper .s.now { border: 2px solid #1e3a68; background: #eef2ff; } .stepper .s.now b { color: #1e3a68; }
.notice { border-radius: 8px; padding: 8px 12px; margin: 6px 0; font-size: 0.86rem; }
.notice.n-conflict { background: #fff8e1; border: 1px solid #f3d27a; color: #6b4e00; }
.notice.n-data { background: #f1f5f9; border: 1px solid #cbd5e1; color: #334155; }
.lead { font-size: 0.95rem; color: #0f1f3d; margin-top: 8px; }
.support { font-size: 0.82rem; color: #64748b; margin-top: 2px; }
.recorded { background: #fff; border: 1px solid #cbd5e1; border-left: 4px solid #1e3a68; border-radius: 8px;
            padding: 8px 12px; margin: 6px 0; font-size: 0.9rem; color: #1e293b; }
.decision-h { font-size: 1.05rem; font-weight: 700; color: #0f1f3d; }
.bar { height: 12px; border-radius: 999px; background: #e2e8f0; margin: 2px 0 10px 0; overflow: hidden; }
.bar > span { display: block; height: 100%; border-radius: 999px; }
.barlabel { font-size: 0.78rem; color: #475569; }
.field-err { color: #b71c1c; font-size: 0.78rem; margin: -6px 0 8px 0; }
.coach-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


def card(label, value, sub="", color="#0f1f3d", tip=""):
    """Summary card; `tip` adds an ⓘ with a hover explanation."""
    t = (f' <span title="{html.escape(tip)}" style="cursor:help">ⓘ</span>' if tip else "")
    st.markdown(f'<div class="card"><div class="label">{label}{t}</div>'
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
@st.cache_data(show_spinner=False)
def demo_squad():
    return pd.read_csv(SQUAD_PATH)


def load_squad():
    """Demo squad (real held-out athlete-weeks) with any weekly data entered in the app on top.
    Entries are shared by every viewer, like the sign-offs."""
    base = demo_squad().copy()
    base["_entered"] = "demo week (real anonymised athlete record)"
    entries = audit.load_entries()
    if entries:
        new = pd.DataFrame(list(entries.values()))
        base = pd.concat([base[~base.player_id.isin(new.player_id)], new], ignore_index=True)
    return base


ss.squad = load_squad()
if "briefings" not in ss:
    ss.briefings = {}      # cache: key -> briefing result (so reruns never call the LLM again)
# Medical decisions are shared by every browser session (SQLite), so the coach sees the doctor's
# decision on any device. Re-read on every run so other sessions' sign-offs appear.
ss.signoffs = audit.load_signoffs()
MODEL_VERSION = (f"LogReg-6f · cut-off {MODEL['threshold']:.4f} · sklearn {MODEL.get('sklearn_version', '?')}")
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
    return decision_state(s, rec_hash(rec)), s


def signoff_label(pid, rec):
    state, s = signoff_state(pid, rec)
    return {"none": "Review pending", "stale": "Superseded – re-sign"}.get(state) or f"Signed · {s['final_status']}"


def risk_tag(res):
    if res["no_recent_data"]:
        return "⚪ Data check"
    return ("🔴 High" if res["risk_level"] == "High" else "🟢 Low") + (" · borderline" if res["borderline"] else "")


def latest_briefing(pid, rec):
    prev = [v for k, v in ss.briefings.items() if k.startswith(f"{pid}|{rec_hash(rec)}|")]
    return prev[-1] if prev else None


def goto_player(pid):
    """Select a player and switch to the Player assessment tab (a tiny browser script, see the end of the file).
    If the browser blocks the script, the player is still selected and a hint says which tab to open."""
    ss["brief_player"] = pid
    ss["_goto_nonce"] = ss.get("_goto_nonce", 0) + 1
    ss["_goto_pending"] = True


jump_to = goto_player
ROLE_WORKSPACE = {"Sports scientist": "Staff workspace", "Club doctor": "Medical review",
                  "Head physiotherapist": "Medical review", "Head coach": "Signed-off decisions only"}


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
    ss.setdefault("pin_fails", 0)
    ss.setdefault("pin_seen", [])
    if role != "Head coach":
        if STAFF_PIN and ss.pin_fails >= MAX_PIN_TRIES:
            staff_ok = False
            st.error("🔒 Too many wrong staff PINs: medical views are locked for this session.")
        elif STAFF_PIN:
            pin = st.text_input("Staff PIN", type="password",
                                help="Medical and sports-science views are PIN-protected. The coach view needs no PIN.")
            staff_ok = bool(pin) and hmac.compare_digest(pin, STAFF_PIN)
            if pin and not staff_ok:
                tag = hashlib.sha256(pin.encode()).hexdigest()[:12]   # never store the attempt itself
                if tag not in ss.pin_seen:
                    ss.pin_seen.append(tag)
                    ss.pin_fails += 1
                    audit.log_event("staff_pin_rejected", actor=reviewer, details={"role": role, "attempt": ss.pin_fails})
                st.error(f"Wrong staff PIN ({ss.pin_fails}/{MAX_PIN_TRIES} attempts).")
        else:
            st.caption("⚠️ No staff PIN set in Secrets, so anyone can open the medical views.")

    if role != "Head coach" and staff_ok:
        st.divider()
        with st.expander("🎬 Demo players", expanded=True):
            st.caption("Click a player to open his assessment.")
            known = set(ss.squad["player_id"])
            for pid_, label_ in DEMO_CASES:
                if pid_ in known:
                    st.button(f"{pid_} · {label_}", key=f"demo_{pid_}", on_click=jump_to, args=(pid_,),
                              use_container_width=True)

    st.divider()
    staff_tools = staff_ok and role != "Head coach"   # the coach (no PIN) never sees AI settings or Reset
    llm_model = MAIN_MODEL if MAIN_MODEL in MODEL_OPTIONS else MODEL_OPTIONS[0]
    if staff_tools:
        if CLIENT is None:
            st.warning("🤖 No Groq key found: briefings use the fixed fallback template.")
        else:
            st.success("🤖 AI briefings on (Groq key loaded)")
        llm_model = st.selectbox("Briefing model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(llm_model),
                                 help="Only models your Groq key can use right now are listed.")
        if CLIENT is not None and PROMPT_GUARD_ON is not None:
            st.caption("🛡️ Prompt Guard 2 classifier: " + ("on" if PROMPT_GUARD_ON else
                       "not available on this key (keyword screen only)"))
        if st.button("🔄 Reset demo", use_container_width=True,
                     help="Clears all sign-offs and data edits (for every viewer) and this browser's briefings. "
                          "Staff only."):
            audit.clear_signoffs()   # shared: clears the demo decisions and data edits for every viewer
            audit.clear_entries()
            for k in ["squad", "briefings", "notes", "brief_player", "signoff_ver", "flash", "_squad_sel"]:
                ss.pop(k, None)
            audit.log_event("demo_reset", actor=f"{reviewer} ({role})")
            st.rerun()
        st.caption(f"Model cut-off {MODEL['threshold']:.4f} · borderline band ±{BAND:.0%}")

# ------------------------------------------------------------------ header
workspace = ROLE_WORKSPACE[role] if (role == "Head coach" or staff_ok) else "Coach view until the staff PIN is entered"
st.markdown(
    '<div class="hero"><div><h1>⚽ AI-Assisted Player Health & Injury Risk</h1>'
    '<p>Decision support for the club medical team</p></div>'
    f'<span class="tag">{ROLE_ICONS[role]} {role.upper()} · {workspace.upper()}</span></div>'
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
        return "background-color:#e2e8f0;color:#1e293b;font-weight:600"
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
        row["_order"] = (0, s["at"]) if state == "done" else (1, p)
        rows.append(row)
    rows = sorted([r for r in rows if r["_order"][0] == 0], key=lambda r: r["_order"][1], reverse=True) + \
        sorted([r for r in rows if r["_order"][0] == 1], key=lambda r: r["_order"][1])
    cdf = pd.DataFrame(rows).drop(columns="_order")
    done = cdf[cdf.Status != "Awaiting medical sign-off"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Signed off", len(done), f"of {len(cdf)} players", "#1e3a68")
    with c2:
        card("Train as planned", int((done.Status == "Train as planned").sum()), "no change", "#1e293b")
    with c3:
        card("Modified load", int(done.Status.str.startswith("Modified").sum()), "medical plan", "#b26a00")
    with c4:
        card("Assessment first", int(done.Status.str.startswith("Medical assessment").sum()),
             "before next session", "#b71c1c")
    st.write("")
    st.dataframe(cdf.style.map(color_status, subset=["Status"]), use_container_width=True, hide_index=True,
                 height=height, column_config={"Player": st.column_config.TextColumn(width="small"),
                                               "Medical note": st.column_config.TextColumn("Note from the medical team", width="medium")})
    st.caption("Signed-off players are listed first, newest first.")


# ================================================================== HEAD COACH VIEW
if role == "Head coach" or not staff_ok:
    if role == "Head coach" and not ss.get("_coach_logged"):
        ss["_coach_logged"] = True     # once per browser session
        audit.log_event("coach_view_opened", actor=reviewer, details={
            "signed_off_visible": sum(signoff_state(r["player_id"], r)[0] == "done" for r, _ in scored)})
    if not staff_ok:
        st.warning("🔒 Enter the staff PIN in the sidebar to open the medical views. Until then you see the coach view.")
    st.subheader("📣 Coach view: this week's signed-off status")
    st.caption("Only outcomes signed off by the medical team appear here. Risk scores and wellness data are not "
               "shown to coaching staff (protocol C-16).")
    render_coach_table(height=620)
    st.stop()

# ================================================================== STAFF VIEWS
# Daily work first; evaluation, audit log and protocol grouped as secondary tools in the last tab.
tabs = st.tabs(["📋 Squad overview", "🩺 Player assessment", "✍️ Weekly data", "👀 Coach view", "📊 Evaluation & records"])

# ------------------------------------------------------------------ 1. squad dashboard
with tabs[0]:
    rows = []
    for rec, res in scored:
        facts = facts_for_protocol(rec, res)
        _, info = retrieve(PROTOCOL, dict(facts))
        state = signoff_state(rec["player_id"], rec)[0]
        flagged = (res["risk_level"] == "High" or res["no_recent_data"] or bool(info["conflict_pairs"]))
        # Review priority: flagged and unsigned first, then borderline, then the rest; signed-off last.
        prio = 3 if state == "done" else (0 if flagged else 1 if res["borderline"] else 2)
        if res["no_recent_data"]:
            kind, why = "q-data", "⚪ Data check · no distance logged for 3 weeks"
        elif info["conflict_pairs"]:
            kind, why = "q-conflict", ("⚠️ Protocol conflict (" + ", ".join(f"{a} vs {b}" for a, b in info["conflict_pairs"])
                                       + ") · review required")
        elif res["risk_level"] == "High":
            kind, why = "q-high", f"🔴 High-risk flag ({res['risk_index']:.1f}×) · review pending"
        else:
            kind, why = "q-border", f"🟠 Borderline score ({res['risk_index']:.1f}×) · review pending"
        if state == "stale":
            why += " · old decision superseded"
        rows.append({
            "_prio": prio, "_review": prio in (0, 1), "_kind": kind, "_why": why,
            "Player": rec["player_id"],
            "Risk": risk_tag(res),
            "Risk index": round(res["risk_index"], 1),
            "Main risk factor": res["reasons"][0] if res["reasons"] else "—",
            "Medical review": signoff_label(rec["player_id"], rec),
            "ACWR": None if res["no_recent_data"] else round(res["features"]["acwr"], 2),
            "Sleep (h)": rec["sleep_hours"],
            "Protocol flags": (("⚠️ " + ", ".join(f"{a} vs {b}" for a, b in info["conflict_pairs"]) + " · ")
                               if info["conflict_pairs"] else "") +
                              (", ".join(c for c in info["rule_clauses"] if c not in ("C-1", "C-2", "C-3", "C-16", "C-17")) or "—"),
        })
    df = pd.DataFrame(rows).sort_values(["_prio", "Risk index"], ascending=[True, False]).reset_index(drop=True)
    n_high = sum(r["risk_level"] == "High" and not r["no_recent_data"] for _, r in scored)
    n_border = sum(r["borderline"] and not r["no_recent_data"] for _, r in scored)
    n_data = sum(r["no_recent_data"] for _, r in scored)
    n_signed = int((df._prio == 3).sum())
    n_pending = int(df._review.sum())
    last = audit.read_events(limit=1)

    # Level 1: page header and four cards
    st.markdown(f'<div class="page-h">Squad overview</div><div class="page-sub">Weekly monitoring · {len(scored)} players · '
                f'{n_signed} signed off this week · last activity {last[0]["Timestamp"] if last else "none yet"}</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Players monitored", len(scored), "this week's entries")
    with c2:
        card("High-risk flags", n_high, f"model output · +{n_border} borderline", "#b71c1c",
             tip="Players whose model score is at or above the cut-off. A flag is a model output, not a medical decision.")
    with c3:
        card("Pending medical reviews", n_pending, "flagged or borderline, no decision yet", "#b26a00",
             tip="Flagged (High, protocol conflict or data check) or borderline players without a sign-off on this week's data.")
    with c4:
        card("Data-quality issues", n_data, "no distance logged for 3 weeks", "#475569",
             tip="The workload ratio cannot be calculated, so the score needs a data check first (protocol C-15).")

    # Level 2: priority queue, grouped by why the player needs a human look
    st.markdown('<div class="sec-h">Players needing attention</div>', unsafe_allow_html=True)
    queue = df[df._review]
    if queue.empty:
        st.markdown('<div class="notice n-data">No player needs attention: every flagged or borderline player has a '
                    'medical decision on this week\'s data.</div>', unsafe_allow_html=True)
    else:
        groups = [("🔴 High-risk flags", "q-high", "model score at or above the cut-off", queue[queue._kind == "q-high"]),
                  ("⚠️ Protocol conflicts", "q-conflict", "clauses point different ways; C-17 applies",
                   queue[queue._kind == "q-conflict"]),
                  ("⚪ Data checks · 🟠 borderline", "q-data", "missing distance data, or a score near the cut-off",
                   queue[queue._kind.isin(["q-data", "q-border"])])]
        qcols = st.columns(3)
        for col, (title, cls, expl, g) in zip(qcols, groups):
            with col:
                st.markdown(f'<div class="queue {cls}"><b>{title}</b> <span class="support">· {len(g)}</span>'
                            f'<div class="qwhy">{expl}</div></div>', unsafe_allow_html=True)
                if g.empty:
                    st.caption("None this week.")
                for q in g.head(5).to_dict("records"):
                    short = {"q-high": f"{q['Risk index']:.1f}×", "q-data": "data check", "q-border": "borderline"}.get(
                        q["_kind"], q["Protocol flags"].split(" · ")[0].replace("⚠️ ", ""))
                    st.button(f"{q['Player']} · {short}" + (" · re-sign" if "superseded" in q["_why"] else "") + "  →",
                              key=f"queue_open_{q['Player']}", on_click=goto_player, args=(q["Player"],),
                              use_container_width=True, help=q["_why"])
                if len(g) > 5:
                    st.caption(f"+ {len(g) - 5} more: use the **Needs attention** filter below.")

    # Level 3: all players
    st.markdown('<div class="sec-h">All players</div>', unsafe_allow_html=True)

    def clear_filters():
        ss["squad_filter"], ss["squad_search"] = "All", ""

    ss.setdefault("squad_filter", "All")   # set here (not via default=) so "Clear filters" can reset it
    f1, f2, f3 = st.columns([2.3, 2, 1.4])
    with f1:
        view = st.segmented_control("Show", ["All", "Needs attention", "Awaiting sign-off"],
                                    key="squad_filter",
                                    help="Needs attention = High risk, borderline, protocol conflict or data check.") or "All"
    with f2:
        search = st.text_input("Search player", placeholder="e.g. P-29", key="squad_search").strip()
    with f3:
        details = st.toggle("Workload & protocol columns", key="squad_details",
                            help="Adds ACWR, sleep and the protocol clauses each player's numbers trigger.")
    shown = df
    if view == "Needs attention":
        shown = df[df.Risk.str.contains("High|borderline|Data check") | df["Protocol flags"].str.contains("⚠️")]
    elif view == "Awaiting sign-off":
        shown = df[~df["Medical review"].str.startswith("Signed")]
    if search:
        shown = shown[shown.Player.str.contains(search, case=False, regex=False)]
    if view != "All" or search:
        a1, a2 = st.columns([4, 1])
        with a1:
            st.caption("Active filters: " + " · ".join(([f"**{view}**"] if view != "All" else []) +
                                                      ([f"search **{search}**"] if search else [])))
        with a2:
            st.button("✕ Clear filters", on_click=clear_filters, key="clear_filters", use_container_width=True)
    cols_shown = ["Player", "Risk", "Risk index", "Main risk factor", "Medical review"] + \
        (["ACWR", "Sleep (h)", "Protocol flags"] if details else [])
    shown = shown[cols_shown].reset_index(drop=True)

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

    def color_review(v):
        return "color:#1e3a68;font-weight:600" if str(v).startswith("Signed") else "color:#b26a00"

    if shown.empty:
        st.markdown('<div class="notice n-data">No players match these filters. Press <b>✕ Clear filters</b> to see '
                    'the full squad.</div>', unsafe_allow_html=True)
        ss["_squad_sel"] = None
    else:
        fmt = {"Risk index": "{:.1f}×"}
        if details:
            fmt.update({"ACWR": lambda x: "—" if pd.isna(x) else f"{x:.2f}", "Sleep (h)": "{:.1f}"})
        styled = shown.style.map(color_risk, subset=["Risk"]).map(color_review, subset=["Medical review"]).format(fmt)
        squad_cols = {
            "Player": st.column_config.TextColumn(width="small"),
            "Risk": st.column_config.TextColumn(width="small", help="Model flag: High / Low, borderline = within "
                                                                    f"±{BAND:.0%} of the cut-off, Data check = no recent distance"),
            "Risk index": st.column_config.Column(width="small", help="Model score ÷ the average athlete-day. 2.0× = twice "
                                                                      "the average. A relative ranking, not a probability."),
            "Main risk factor": st.column_config.TextColumn(width="large"),
            "Medical review": st.column_config.TextColumn(width="medium", help="The human decision, separate from the model flag"),
            "ACWR": st.column_config.Column(width="small", help="This week's distance ÷ average of the last 3 weeks"),
            "Sleep (h)": st.column_config.Column(width="small", help="Protocol context only (C-8), not used by the model"),
            "Protocol flags": st.column_config.TextColumn(width="medium",
                                                          help="Extra protocol clauses triggered by this player's numbers; ⚠️ = clauses that conflict"),
        }
        table_kw = dict(use_container_width=True, hide_index=True, height=min(560, 38 + 35 * len(shown)),
                        column_config={k: v for k, v in squad_cols.items() if k in cols_shown})
        try:   # click a row to open the player
            event = st.dataframe(styled, on_select="rerun", selection_mode="single-row",
                                 key=f"squad_table_{view}_{search}_{details}", **table_kw)
            picked_rows = list(event.selection.rows)
        except Exception:  # noqa: BLE001  (row selection unavailable: plain table, use the queue or Player tab)
            st.dataframe(styled, **table_kw)
            picked_rows = []
        if picked_rows and picked_rows[0] < len(shown):
            picked = shown.iloc[picked_rows[0]]["Player"]
            if ss.get("_squad_sel") != picked:      # only when the click changes, so the Player tab can still switch
                ss["_squad_sel"] = picked
                goto_player(picked)
            st.caption(f"**{picked}** selected — shown in **🩺 Player assessment**.")
        else:
            ss["_squad_sel"] = None
    st.caption(f"Showing {len(shown)} of {len(df)} players, sorted by review priority (flagged and unsigned first, "
               "signed-off last). Click a row to open the player. Squad = 30 real athlete-weeks from held-out test "
               "athletes (anonymised P-01…P-30); sleep values are illustrative and only drive protocol clause C-8.")
    with st.expander("ℹ️ How a score becomes this week's flag"):
        st.markdown(DAYS_VS_WEEKS)
    st.download_button("📥 Download squad table (CSV)", df.drop(columns=["_prio", "_review", "_kind", "_why"]).to_csv(index=False),
                       "squad_risk.csv", "text/csv")

# ------------------------------------------------------------------ 2. player page
with tabs[1]:
    order = [r["player_id"] for r, _ in sorted(scored, key=lambda t: -t[1]["probability"])]
    if ss.get("brief_player") not in order:
        ss.pop("brief_player", None)
    res_by_id = {r["player_id"]: s for r, s in scored}

    def player_label(p):
        return f"{p}  ·  {risk_tag(res_by_id[p])}  ·  {signoff_label(p, get_rec(p))}"

    pid = st.selectbox("Player (highest model risk first)", order, key="brief_player", format_func=player_label)
    rec = get_rec(pid)
    res = score(MODEL, rec, BAND)
    facts = facts_for_protocol(rec, res)
    _, rule_info = retrieve(PROTOCOL, dict(facts))
    state, sign = signoff_state(pid, rec)
    existing_briefing = latest_briefing(pid, rec)
    esc = html.escape
    lvl = res["risk_level"]
    aid = rec_hash(rec)

    # ---- player header: who, model flag, review state and next step first; supporting numbers below ----
    with st.container(border=True):
        flag = ('<span class="pill pill-data">Data check needed</span>' if res["no_recent_data"] else
                (f'<span class="pill pill-high">High-risk flag (model)</span>' if lvl == "High" else
                 '<span class="pill pill-low">Low risk (model)</span>'))
        if res["borderline"] and not res["no_recent_data"]:
            flag += '<span class="pill pill-border">borderline</span>'
        review = {"done": f'<span class="pill pill-done">Signed · {esc(sign["final_status"]) if sign else ""}</span>',
                  "stale": '<span class="pill pill-border">Decision superseded – re-sign</span>',
                  "none": '<span class="pill pill-wait">Medical review pending</span>'}[state]
        st.markdown(f'<div class="support">PLAYER ASSESSMENT · this week · assessment <span class="mono">{aid}</span></div>'
                    f'<span class="player-id">{pid}</span>{flag}{review}', unsafe_allow_html=True)

        # guided workflow
        s2 = existing_briefing is not None
        s3 = state == "done"
        steps = [("1 · Review player data", "⚠ no distance for 3 weeks — check" if res["no_recent_data"] else "weekly inputs complete", True),
                 ("2 · Generate briefing", "AI draft with protocol evidence", s2),
                 ("3 · Medical sign-off", "doctor / physio decision" if state != "stale" else "re-sign on the new data", s3),
                 ("4 · Coach view", "signed-off outcome only", s3)]
        current = next((i for i, (_, _, d) in enumerate(steps) if not d), None)
        st.markdown('<div class="stepper">' + "".join(
            f'<div class="s {"done" if d else ("now" if i == current else "")}"><b>{"✓ " if d else ""}{t}</b>{sub_}</div>'
            for i, (t, sub_, d) in enumerate(steps)) + "</div>", unsafe_allow_html=True)
        nxt = {None: f"Done: <b>{esc(sign['final_status']) if sign else ''}</b> is visible to the head coach.",
               1: "Next step: generate the protocol briefing in <b>① Briefing</b> below.",
               2: "Next step: record the medical decision in <b>② Medical sign-off</b> below."}.get(current, "")
        st.markdown(f'<div class="next">{nxt}</div>', unsafe_allow_html=True)

        h1, h2, h3 = st.columns(3)
        with h1:
            card("Risk index", f"{res['risk_index']:.1f}×",
                 f"vs average athlete-day · higher than {min(res['percentile'], 99)}% of training days",
                 "#b71c1c" if lvl == "High" else "#0f1f3d",
                 tip=f"Model score ÷ average score ({MODEL['base_rate']:.2%}). 2.0× = twice the average athlete-day. "
                     "A relative ranking, not a probability of injury.")
        with h2:
            card("Workload ratio (ACWR)", "—" if res["no_recent_data"] else f"{res['features']['acwr']:.2f}",
                 "this week ÷ 3-week average",
                 tip="Acute:chronic workload ratio. Above 1.5 = workload spike (C-5); below 0.8 = under-loading (C-7).")
        with h3:
            card("Sleep", f"{rec['sleep_hours']:g} h", "per night · protocol context only",
                 "#b26a00" if rec["sleep_hours"] < 7 else "#0f1f3d",
                 tip="Not used by the model (no sleep in the training data). Below 7 h triggers protocol clause C-8.")

        reasons = res["reasons"]
        if reasons:
            st.markdown(f'<div class="lead">Why the model rates him {lvl.lower()}: main factor '
                        f'<b style="color:{"#b71c1c" if lvl == "High" else "#0f1f3d"}">{esc(reasons[0])}</b></div>'
                        + (f'<div class="support">Also contributing: {esc(" · ".join(reasons[1:]))}</div>' if reasons[1:] else ""),
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="lead">Why the model rates him low: no factor pushes his score above average.</div>',
                        unsafe_allow_html=True)
        extra_cl = [c for c in rule_info["rule_clauses"] if c not in ("C-1", "C-2", "C-3", "C-16", "C-17")]
        st.markdown(f'<div class="support">Model factors come from the last 3 weeks of training. Protocol context, not '
                    f'used by the model: sleep {rec["sleep_hours"]:g} h'
                    + (f' · clauses triggered by his numbers: {", ".join(extra_cl)}' if extra_cl else "") + "</div>",
                    unsafe_allow_html=True)
        # notices: most consequential first, each with its own treatment
        if rule_info["conflict_pairs"]:
            pairs = ", ".join(f"{a} vs {b}" for a, b in rule_info["conflict_pairs"])
            st.markdown(f'<div class="notice n-conflict">⚠️ <b>Protocol conflict ({pairs}).</b> Clause C-17 applies: no '
                        'load increase and a medical review within 24 hours until the club doctor decides.</div>',
                        unsafe_allow_html=True)
        if res["no_recent_data"]:
            st.markdown('<div class="notice n-data">ⓘ <b>Data check:</b> no training distance logged in the last 3 weeks, '
                        'so the workload ratio cannot be calculated. Correct the numbers in <b>✍️ Weekly data</b> before '
                        'relying on this score (protocol C-15).</div>', unsafe_allow_html=True)
        st.caption((f"Model score {res['probability']:.1%} per athlete-day vs {MODEL['base_rate']:.2%} on average; High "
                    f"from {res['threshold']:.2%}. Use it to rank players, not as a forecast for one player. "
                    if not res["no_recent_data"] else "") +
                   f"Data: {rec.get('_entered') or 'entered in the app'}. The assessment ID changes whenever the numbers "
                   "change; the sign-off is tied to it.")

    sub = st.tabs(["① Briefing", "② Medical sign-off", "🔮 What-if", "🔢 This week's data"])

    # ---- briefing ----
    with sub[0]:
        n1, n2 = st.columns(2)
        with n1:
            notes = st.text_area("Staff notes (optional, treated as data, never as instructions)",
                                 value=ss.notes.get(pid, ""), max_chars=600, key=f"notes_{pid}", height=96)
        with n2:
            question = st.text_area("Question for the protocol (optional)", max_chars=300, key=f"q_{pid}", height=96,
                                    placeholder="e.g. How much can we reduce his load next week?")
        go = st.button("✨ Generate protocol briefing", type="primary")
        st.caption("Retrieves the matching protocol clauses and drafts a briefing (a few seconds). Your notes and "
                   "question stay here if anything fails.")
        key = f"{pid}|{aid}|{notes}|{question}|{llm_model}"

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
                             f"({why[0][:90]}). Nothing was sent to the AI. Rephrase it as plain information.")
                audit.log_event("input_blocked", pid, reviewer, {"blocked": blocked, "notes": notes, "question": question})
            elif key in ss.briefings:
                st.info("This exact briefing was already generated (cached, no new AI call).")
            else:
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
                    "generated_by": out["generated_by"], "attempts": out["attempts"], "model_used": out.get("model_used"),
                    "guardrail_problems": out["problems"], "clauses": [c[0] for c in out["clauses"]],
                    "risk_level": res["risk_level"], "risk_index": round(res["risk_index"], 2),
                    "assessment_id": aid, "question": question, "notes": notes})
                st.rerun()   # refresh the header so the guided steps move on to sign-off

        out = ss.briefings.get(key) or latest_briefing(pid, rec)
        if out:
            b = out["briefing"]
            by_ai = out["generated_by"] == "LLM (Groq)"
            if not by_ai:
                why_ = out["problems"][-1] if out["problems"] else "unknown"
                if why_ == "Failed validation twice" and len(out["problems"]) > 1:
                    why_ += f", e.g. {out['problems'][0][:80]}"
                vline = (f"⚠️ <b>Validation:</b> the AI answer was not used ({esc(why_)}), so the fixed template "
                         "built from the retrieved clauses is shown instead.")
            elif out["attempts"] == 1 and not out["problems"]:
                vline = "✓ <b>Validation:</b> passed every output check on the first attempt."
            else:
                vline = (f"🔁 <b>Validation:</b> the first answer failed {len(out['problems'])} check(s) "
                         f"({esc(out['problems'][0][:90])}); the AI was re-prompted once and the corrected answer passed.")
            inf = out["info"]
            clause_map = {cid: (title, text) for cid, title, text in out["clauses"]}

            def source_tag(cid):
                tags = (["matched by his numbers"] if cid in inf["rule_clauses"] else []) + \
                       (["matched by the question"] if cid in inf["question_clauses"] else [])
                return " · ".join(tags) or "retrieved"

            not_cov = bool(b.get("answer_to_question")) and NOT_COVERED.lower()[:20] in b["answer_to_question"].lower()
            with st.container(border=True):
                st.markdown(f'<span class="draft">{"🤖 AI-generated draft" if by_ai else "📋 Fixed template"} — '
                            f'not a medical decision</span><div class="vstat">{vline}</div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-h" style="margin-top:4px">Summary</div>', unsafe_allow_html=True)
                st.markdown(esc(b["summary"]))
                if b.get("key_factors"):
                    st.markdown('<div class="support"><b>Key factors:</b> ' + " · ".join(esc(k) for k in b["key_factors"])
                                + "</div>", unsafe_allow_html=True)
                st.markdown('<div class="sec-h">Protocol evidence</div><div class="support">Open <i>Source</i> to '
                            'check the wording against the clause.</div>', unsafe_allow_html=True)
                for a in b["protocol_actions"]:
                    cid = a["clause"]
                    title, text = clause_map.get(cid, ("", "clause text not found"))
                    st.markdown(f'<div class="clause"><b class="mono">{esc(cid)}</b> <b>{esc(title)}</b> — {esc(a["action"])}'
                                f'<details><summary>Source text · {source_tag(cid)}</summary>'
                                f'<div class="src">{esc(text)}</div></details></div>', unsafe_allow_html=True)
                if b.get("answer_to_question") and not not_cov:
                    st.markdown(f"**Answer to your question:** {esc(b['answer_to_question'])}")
                st.markdown('<div class="sec-h">Unresolved issues</div>', unsafe_allow_html=True)
                issues = [f"⚠️ <b>Conflicting clauses:</b> {esc(' '.join(b['conflicts']))}"] if b.get("conflicts") else []
                if not_cov:
                    issues.append(f"ⓘ <b>Your question:</b> {esc(b['answer_to_question'])}")
                if issues:
                    for it in issues:
                        st.markdown(f'<div class="notice {"n-conflict" if it.startswith("⚠️") else "n-data"}">{it}</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown('<div class="support">None: no conflicting clauses and nothing outside the protocol.</div>',
                                unsafe_allow_html=True)
                st.markdown(f"**For the medical team:** {esc(b['for_medical_team'])}")
            with st.expander("🔍 Evidence and validation details"):
                st.markdown(f"**Clauses matched by the player's data:** {', '.join(inf['rule_clauses']) or '—'}  \n"
                            f"**Clauses matched by the question:** {', '.join(inf['question_clauses']) or '—'} "
                            f"(similarity {inf['question_similarities'] or '—'})")
                st.markdown("**Retrieved clause text**")
                for cid, title, text in out["clauses"]:
                    st.markdown(f"- **{cid} {title}** — {text}")
                st.markdown("**Validation checks**")
                if out["problems"]:
                    for p in out["problems"]:
                        st.markdown(f"- {p}")
                    st.caption("Checks that failed along the way; the answer was re-prompted or replaced by the template.")
                else:
                    st.markdown("- All output checks passed on the first attempt.")
                st.caption(f"Produced by {out['generated_by']}" + (f" ({out['model_used']})" if out.get("model_used") else "")
                           + f" · {out['attempts']} attempt(s) · {out['created']}")
            if state != "done":
                st.caption("Next: open **② Medical sign-off** above.")
        else:
            st.info("Press **✨ Generate protocol briefing** to retrieve the relevant protocol clauses and draft a "
                    "briefing for the medical team.")

    # ---- sign-off (human checkpoint) ----
    with sub[1]:
        out = latest_briefing(pid, rec)
        needs_caution = res["risk_level"] == "High" or res["no_recent_data"] or bool(rule_info["conflict_pairs"])
        with st.container(border=True):
            b_status = ("none yet" if out is None else
                        ("AI draft, passed the checks" if out["generated_by"] == "LLM (Groq)" else "fixed template (AI not used)"))
            st.markdown(f'<div class="decision-h">🩺 Medical decision · {pid}</div>'
                        f'<div class="support">Assessment <span class="mono">{aid}</span> · model: '
                        f'{"data check" if res["no_recent_data"] else lvl + "-risk flag"} ({res["risk_index"]:.1f}×) · '
                        f'data: {esc(str(rec.get("_entered") or "entered in the app"))} · briefing: {b_status}</div>',
                        unsafe_allow_html=True)
            if state == "done":
                st.markdown(f'<div class="recorded">Decision recorded: <b>{esc(sign["final_status"])}</b> '
                            f'({esc(sign["decision"])}) · by {esc(sign["by"])} ({esc(sign["role"])}) · {esc(sign["at"])}'
                            + (f' · decision <span class="mono">{esc(sign["decision_id"])}</span>' if sign.get("decision_id") else "")
                            + f'<br>Message to the coach: <i>{esc(sign["coach_note"])}</i></div>', unsafe_allow_html=True)
                if sign.get("reason"):
                    st.caption(f"Reason (staff only): {sign['reason']}")
            elif state == "stale":
                st.markdown(f'<div class="notice n-conflict"><b>Superseded:</b> the last decision ({esc(sign["final_status"])}, '
                            f'by {esc(sign["by"])} at {esc(sign["at"])}) was made on different weekly data, so it no longer '
                            'applies and the coach no longer sees it. Please re-sign.</div>', unsafe_allow_html=True)
            ss.setdefault("signoff_fails", 0)
            if role not in SIGNOFF_ROLES:
                st.info(f"🔒 Only the club doctor or head physiotherapist can sign off. You are signed in as "
                        f"**{role}**: switch role in the sidebar.")
            elif out is None:
                st.info("Generate the briefing first (**① Briefing**), so the decision is based on it.")
            elif SIGNOFF_PIN and ss.signoff_fails >= MAX_PIN_TRIES:
                st.error("🔒 Too many wrong sign-off PINs: signing is locked for this session.")
            else:
                box = st.expander("✏️ Change this decision", expanded=False) if state == "done" else st.container()
                with box:
                    ss.setdefault("signoff_ver", {})
                    ver = ss.signoff_ver.get(pid, 0)   # new key after each sign-off, so the form starts empty
                    with st.form(f"signoff_{pid}_{ver}"):
                        decision = st.radio("Your decision on the briefing", ["Approve", "Modify", "Override"], horizontal=True,
                                            help="Approve = accept as drafted. Modify = change the plan. "
                                                 "Override = disagree with the model's risk level.")
                        # No pre-selected status, so the reviewer makes an active choice (reduces automation bias).
                        final_status = st.selectbox("Training status for this week", STATUS_OPTIONS,
                                                    index=None, placeholder="Choose a status...")
                        coach_msg = st.selectbox("Message the head coach will see (protocol C-16: no medical details)",
                                                 COACH_MESSAGES, index=None, placeholder="Choose a message...")
                        extra = st.text_input("Optional extra line for the coach (logistics only, e.g. 'Joins the group "
                                              "from Thursday 10:30')", max_chars=160)
                        reason = st.text_area("Reason (staff only; required for Modify, Override, or 'Train as planned' "
                                              "when the model or protocol flags the player)", max_chars=400)
                        confirmed = st.checkbox("I have read the briefing and its source clauses. This decision is mine, "
                                                "not the model's.")
                        entered_pin = st.text_input("Sign-off PIN (club doctor / head physio only)", type="password") \
                            if SIGNOFF_PIN else None
                        st.caption(f"Signing as **{reviewer}** ({role}). The decision is saved for every device and logged.")
                        submitted = st.form_submit_button("Record medical decision", type="primary")
                    if submitted:
                        note_problems = screen_coach_note(extra, pid, player_ids)
                        pin_ok = (not SIGNOFF_PIN) or bool(entered_pin and hmac.compare_digest(entered_pin, SIGNOFF_PIN))
                        problem = signoff_problem(role, pin_ok, confirmed, final_status, coach_msg, decision, reason,
                                                  needs_caution, note_problems)
                        if problem and problem[0] == "pin":
                            ss.signoff_fails += 1
                            st.error(f"{problem[1]} ({ss.signoff_fails}/{MAX_PIN_TRIES} attempts)")
                            audit.log_event("signoff_rejected_pin", pid, f"{reviewer} ({role})", {"attempt": ss.signoff_fails})
                        elif problem:
                            st.error(problem[1])
                            if problem[0] == "note":
                                audit.log_event("coach_note_blocked", pid, f"{reviewer} ({role})", {"problems": note_problems})
                        else:
                            coach_note = coach_msg + (f" {extra.strip()}" if extra.strip() else "")
                            try:
                                stored = audit.save_signoff(pid, {
                                    "decision": decision, "final_status": final_status, "coach_note": coach_note,
                                    "reason": reason, "by": reviewer, "role": role, "at": now_ist(),
                                    "data_hash": aid, "assessment_id": aid, "model_version": MODEL_VERSION,
                                    "model_level": res["risk_level"], "model_risk_index": round(res["risk_index"], 2),
                                    "briefing_by": out["generated_by"], "briefing_model": out.get("model_used"),
                                    "against_model": needs_caution and final_status == "Train as planned",
                                    "supersedes": sign.get("decision_id") if sign else None})
                            except Exception as e:  # noqa: BLE001
                                st.error(f"The decision could not be saved ({str(e)[:80]}). Nothing was signed; your "
                                         "entries are still in the form, so press the button again.")
                            else:
                                ss.signoff_ver[pid] = ver + 1
                                audit.log_event("signoff", pid, f"{reviewer} ({role})", stored)
                                st.rerun()

    # ---- what-if ----
    with sub[2]:
        wk = {"cut": f"wi_cut_{pid}", "hcut": f"wi_hcut_{pid}", "rest": f"wi_rest_{pid}"}

        def reset_what_if(keys=wk):
            for k_ in keys.values():
                ss[k_] = 0

        for k_, v_ in (("cut", 20), ("hcut", 30), ("rest", 1)):   # suggested starting plan
            ss.setdefault(wk[k_], v_)
        same0 = what_if(MODEL, rec, 0, 0, 0)
        w1, w2 = st.columns([1, 1.2])
        with w1:
            st.markdown('<div class="sec-h" style="margin-top:0">Plan for next week</div>', unsafe_allow_html=True)
            cut = st.slider("Reduce total distance by (%)", 0, 50, step=5, key=wk["cut"],
                            help=f"Of next week's planned distance (assumed equal to this week: "
                                 f"{same0['inputs']['km_this_week']:.1f} km). 20% = {0.2 * same0['inputs']['km_this_week']:.1f} km less.")
            hcut = st.slider("Reduce high-intensity distance by (%)", 0, 100, step=10, key=wk["hcut"],
                             help=f"Of this week's high-intensity distance ({same0['inputs']['hi_km']:.1f} km).")
            extra = st.slider("Add rest days", 0, 2, step=1, key=wk["rest"], help="Extra days with no training next week.")
            st.button("↺ Reset to baseline (no change)", on_click=reset_what_if, key=f"wi_reset_{pid}")
        same = same0
        plan = what_if(MODEL, rec, cut, extra, hcut)
        with w2:
            st.markdown('<div class="sec-h" style="margin-top:0">Scenario comparison</div>', unsafe_allow_html=True)
            v1, v2 = st.columns(2)
            with v1:
                card("Baseline (same load)", f"{same['risk_index']:.2f}×",
                     "High-risk flag" if same["risk_level"] == "High" else "Low risk")
            with v2:
                card("With this plan", f"{plan['risk_index']:.2f}×",
                     ("High-risk flag" if plan["risk_level"] == "High" else "Low risk") +
                     f" · {plan['risk_index'] - same['risk_index']:+.2f}× vs baseline")
            thr_idx = MODEL["threshold"] / MODEL["base_rate"]
            scale = max(same["risk_index"], plan["risk_index"], thr_idx) * 1.15
            st.markdown(
                f'<div class="barlabel">Baseline · {same["risk_index"]:.2f}×</div><div class="bar"><span style="width:'
                f'{same["risk_index"] / scale * 100:.0f}%;background:#1e3a68"></span></div>'
                f'<div class="barlabel">With this plan · {plan["risk_index"]:.2f}×</div><div class="bar"><span style="width:'
                f'{plan["risk_index"] / scale * 100:.0f}%;background:#d97706"></span></div>'
                f'<div class="support">Same scale for both bars. High-risk flag from {thr_idx:.2f}×.</div>',
                unsafe_allow_html=True)
        si, pi = same["inputs"], plan["inputs"]
        cmp_rows = [
            ("Distance next week (km)", si["km_this_week"], pi["km_this_week"], "{:.1f}"),
            ("High-intensity distance (km)", si["hi_km"], pi["hi_km"], "{:.1f}"),
            ("Rest days", si["rest_days"], pi["rest_days"], "{:.0f}"),
            ("Workload ratio (ACWR)", same["features"]["acwr"], plan["features"]["acwr"], "{:.2f}"),
            ("Risk index (× average)", same["risk_index"], plan["risk_index"], "{:.2f}"),
            ("Model score per athlete-day (%)", same["probability"] * 100, plan["probability"] * 100, "{:.2f}"),
        ]
        st.dataframe(pd.DataFrame([{"": n, "Baseline (same load)": f.format(x), "This plan": f.format(y),
                                    "Difference": (f.replace("{:", "{:+") + (" pp" if "(%)" in n else "")).format(y - x)}
                                   for n, x, y, f in cmp_rows]), hide_index=True, use_container_width=True)
        if plan["features"]["acwr"] < 0.8 <= same["features"]["acwr"]:
            st.markdown('<div class="notice n-conflict">⚠️ This plan pushes the workload ratio below 0.8, which protocol '
                        'clause C-7 treats as under-loading.</div>', unsafe_allow_html=True)
        st.caption("**What this shows:** how the model's score moves when the numbers change — not proof that the plan "
                   "prevents injury. Both columns shift the 3-week window one week forward. pp = percentage points of "
                   "the model's per-day score, not a change in this player's chance of injury. Cutting load too far "
                   "also raises risk (C-7). Any plan still needs medical sign-off.")

    # ---- this week's data ----
    with sub[3]:
        st.dataframe(pd.DataFrame([(lbl, rec[k]) for k, lbl, *_ in INPUT_FIELDS], columns=["Field", "Value"]),
                     hide_index=True, use_container_width=True)
        st.caption(f"Source: {rec.get('_entered') or 'entered in the app'}. To change these numbers, use the "
                   "**✍️ Weekly data** tab; new data supersedes the player's sign-off.")

# ------------------------------------------------------------------ 3. data entry
FIELD_GROUPS = [
    ("Training load", ["km_this_week", "km_last_week", "km_two_weeks_ago", "hi_km"]),
    ("Recovery & wellness", ["rest_days", "strength_sessions", "exertion", "sleep_hours"]),
    ("Injury history", ["prior_injury_12m"]),
]
FIELD_INFO = {k: (label, lo, hi, default, step) for k, label, lo, hi, default, step in INPUT_FIELDS}

FIELD_HELP = {
    "km_this_week": "GPS total distance this week, in km (0–250).",
    "km_last_week": "Total distance last week, in km (0–250). Used for the workload ratio.",
    "km_two_weeks_ago": "Total distance two weeks ago, in km (0–250). Used for the workload ratio.",
    "hi_km": "High-intensity / sprint distance this week, in km (0–80). Cannot exceed this week's total.",
    "rest_days": "Days with no training this week (0–7).",
    "strength_sessions": "Gym / strength sessions this week (0–9).",
    "exertion": "Average session rating of perceived exertion this week, 0 (rest) to 10 (maximal).",
    "sleep_hours": "Average sleep per night, in hours (0–14). Protocol clause C-8 only; not used by the model.",
    "prior_injury_12m": "Any injury that caused missed training or matches in the past 12 months.",
}


def field_errors(errs):
    """Map validate_inputs() messages to their field, so each error is shown next to its input."""
    out, other = {}, []
    labels = {label: k for k, label, *_ in INPUT_FIELDS}
    for e in errs:
        lab = next((lb for lb in labels if e.startswith(lb + ":")), None)
        if lab:
            out[labels[lab]] = ("choose No or Yes" if labels[lab] == "prior_injury_12m" and "missing" in e
                                else e[len(lab) + 1:].strip())
        elif e.startswith("High-intensity distance cannot"):
            out["hi_km"] = "cannot be more than the total distance this week"
        elif e.startswith("Player ID") or "already exists" in e:
            out["player_id"] = e
        else:
            other.append(e)
    if other:
        out["_other"] = "; ".join(other)
    return out


with tabs[2]:
    st.markdown('<div class="page-h">Weekly data</div><div class="page-sub">Enter one player\'s week or import the '
                'squad CSV. Every field is range-checked before scoring; new numbers create a new assessment and '
                'supersede any earlier sign-off.</div>', unsafe_allow_html=True)
    if ss.get("entry_flash"):
        st.success(ss.pop("entry_flash"))
        fpid = ss.pop("entry_flash_pid", None)
        if fpid:
            st.button(f"Open {fpid} in Player assessment →", key="entry_open", on_click=goto_player, args=(fpid,))
    mode = st.segmented_control("Entry mode", ["One player", "Whole squad (CSV)"], default="One player",
                                key="entry_mode") or "One player"
    if mode == "One player":
        existing = ["➕ New player"] + player_ids
        who = st.selectbox("Player to add or update", existing, key="entry_player")
        base = get_rec(who) if who != "➕ New player" else {}   # a new player starts with empty fields
        if ss.get("entry_errs_for") != who:
            ss["entry_errs"], ss["entry_errs_for"] = {}, who
        ferr = ss.get("entry_errs", {})
        if ferr:
            st.error(f"Not saved: {len([k for k in ferr if k != '_other'])} field(s) need fixing — see the messages "
                     "under each field." + (f" Also: {ferr['_other']}" if "_other" in ferr else ""))

        def show_err(k):
            if k in ferr:
                st.markdown(f'<div class="field-err">⚠ {html.escape(ferr[k])}</div>', unsafe_allow_html=True)

        with st.form("entry"):
            st.markdown('<div class="sec-h" style="margin-top:0">1 · Player</div>', unsafe_allow_html=True)
            new_id = st.text_input("Player ID", value=who if who != "➕ New player" else f"P-{len(player_ids) + 1:02d}",
                                   help="Format P- followed by 2–3 digits, e.g. P-31.")
            show_err("player_id")
            vals = {}
            for gi, (title, keys) in enumerate(FIELD_GROUPS, start=2):
                st.markdown(f'<div class="sec-h">{gi} · {title}</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, k in enumerate(keys):
                    label, lo, hi, default, step = FIELD_INFO[k]
                    v = base.get(k) if base else None
                    v = None if v is None or pd.isna(v) else v
                    with cols[i % 2]:
                        if k == "prior_injury_12m":
                            choice = st.radio("Time-loss injury in the past 12 months?", ["No", "Yes"],
                                              index=None if v is None else int(float(v)), horizontal=True,
                                              key=f"in_{who}_{k}", help=FIELD_HELP[k])
                            vals[k] = None if choice is None else int(choice == "Yes")
                        elif isinstance(step, int):
                            vals[k] = st.number_input(label, value=None if v is None else int(v), step=step,
                                                      placeholder=f"{lo}–{hi}", key=f"in_{who}_{k}", help=FIELD_HELP[k])
                        else:
                            vals[k] = st.number_input(label, value=None if v is None else float(v), step=step,
                                                      placeholder=f"{lo:g}–{hi:g}", key=f"in_{who}_{k}", help=FIELD_HELP[k])
                        show_err(k)
            st.caption("All fields are required. Values outside the allowed range are flagged next to the field.")
            save = st.form_submit_button("Validate & save", type="primary")
        if save:
            errs = validate_inputs(vals)
            if not re.fullmatch(r"P-\d{2,3}", new_id.strip()):
                errs.append("Player ID must look like P-31")
            elif new_id.strip() != who and new_id.strip() in player_ids:
                errs.append(f"{new_id.strip()} already exists: pick it from the list to update it instead")
            if errs:
                ss["entry_errs"] = field_errors(errs)
                audit.log_event("entry_rejected", new_id, reviewer, {"errors": errs})
                st.rerun()      # show each message next to its field (the typed values are kept)
            else:
                row = {"player_id": new_id.strip(), **vals}
                audit.save_entry(row["player_id"], row, by=f"{reviewer} ({role})")
                audit.log_event("entry_saved", row["player_id"], reviewer, {**row, "assessment": rec_hash(row)})
                ss["entry_errs"] = {}
                ss["entry_flash"] = (f"Saved {row['player_id']} (assessment {rec_hash(row)}). Model flag: "
                                     f"{score(MODEL, row, BAND)['risk_level']}. Next: brief and sign off.")
                ss["entry_flash_pid"] = row["player_id"]
                st.rerun()      # so the squad table and player list show the new numbers
    else:
        cols_needed = ["player_id"] + [k for k, *_ in INPUT_FIELDS]
        u1, u2 = st.columns([3, 1])
        with u1:
            up = st.file_uploader("Upload this week's squad CSV", type=["csv"],
                                  help="One row per player, with the columns in the template.")
        with u2:
            st.write("")
            st.download_button("📄 CSV template", ss.squad[cols_needed].head(3).to_csv(index=False),
                               "weekly_template.csv", "text/csv", use_container_width=True)
        st.caption("Columns: " + ", ".join(cols_needed))
        if up is not None:
            try:
                new = pd.read_csv(up)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the CSV ({str(e)[:120]}). Save it as CSV (comma-separated) and try again.")
                new = None
            if new is not None:
                miss = [c for c in cols_needed if c not in new.columns]
                if miss:
                    st.error("Missing columns: " + ", ".join(miss) + ". Download the template to see the expected headers.")
                else:
                    good, bad = [], []
                    dupes = set(new.player_id[new.player_id.duplicated()].astype(str))
                    for r in new[cols_needed].to_dict("records"):
                        errs = validate_inputs(r)
                        if not re.fullmatch(r"P-\d{2,3}", str(r.get("player_id", "")).strip()):
                            errs.append("player_id must look like P-31")
                        if str(r.get("player_id")) in dupes:
                            errs.append("player_id appears more than once in the file")
                        (bad if errs else good).append((r, errs))
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        card("Rows in file", len(new))
                    with k2:
                        card("Valid", len(good), "ready to load", "#1e3a68")
                    with k3:
                        card("Rejected", len(bad), "see the list below", "#b71c1c" if bad else "#1e3a68")
                    if bad:
                        bad_df = pd.DataFrame([{**{c_: r.get(c_) for c_ in cols_needed}, "problems": "; ".join(e)}
                                               for r, e in bad])
                        st.dataframe(bad_df[["player_id", "problems"]], hide_index=True, use_container_width=True)
                        st.download_button("📥 Download rejected rows (CSV, with the problems)", bad_df.to_csv(index=False),
                                           "rejected_rows.csv", "text/csv")
                    if good and st.button(f"Load {len(good)} valid row(s) into the squad", type="primary"):
                        gdf = pd.DataFrame([r for r, _ in good])
                        for r in gdf.to_dict("records"):
                            audit.save_entry(str(r["player_id"]), r, by=f"{reviewer} ({role})")
                        audit.log_event("csv_upload", "", reviewer, {"rows_loaded": len(gdf), "rows_rejected": len(bad)})
                        ss["entry_flash"] = (f"Loaded {len(gdf)} rows ({len(bad)} rejected). Any earlier sign-off made on "
                                             "different numbers is now superseded.")
                        st.rerun()

# ------------------------------------------------------------------ 4. coach view preview
with tabs[3]:
    st.markdown('<div class="page-h">Coach view (preview)</div><div class="page-sub">Exactly what the head coach sees '
                'when the role is set to <b>Head coach</b>: only signed-off status and the message chosen by the '
                'medical team — no scores, factors or wellness data (protocol C-16).</div>', unsafe_allow_html=True)
    render_coach_table(height=520)

# ------------------------------------------------------------------ 5. evaluation
@st.cache_data(ttl=900, show_spinner=False)
def get_acceptance(protocol_raw, eval_stamp):
    """Acceptance suite + failure modes, re-computed when the protocol or the evaluation file changes."""
    squad0 = demo_squad()
    return (acceptance_suite(MODEL, PROTOCOL, squad0, BAND, load_json(METRICS_PATH), load_json(EVAL_PATH), rec_hash),
            failure_mode_tests(MODEL, PROTOCOL, squad0, BAND))


with tabs[4]:
    m = load_json(METRICS_PATH)
    ev = load_json(EVAL_PATH)
    st.markdown('<div class="page-h">Evaluation & records</div><div class="page-sub">Model results, AI safety tests, '
                'the audit log and the club protocol.</div>', unsafe_allow_html=True)
    etabs = st.tabs(["🎯 Risk model", "💰 Cost of errors", "🤖 AI & guardrails", "✅ Acceptance tests",
                     "🗂️ Audit log", "📜 Protocol", "⚠️ Limitations"])

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
                card("Days flagged", f"{lr['flag_rate']:.0%}", "≈ player-weeks when checked weekly")
            with k4:
                card("Lift over base rate", f"{lr['lift_vs_base_rate']:.1f}×", "injury rate in flagged days vs average")
            st.write("")
            st.markdown(f"**Data:** {d['name']} — {d['licence']} licence (DOI 10.34894/UWU9PV). {d['athletes']} real athletes, "
                        f"{d['rows']:,} athlete-days, {d['injuries']} injuries. Row = {d['row_definition'].lower()}; "
                        f"label = {d['label'].lower()}. **Test:** {d['test_athletes']} athletes the model never saw "
                        f"({d['test_rows']:,} rows, {d['test_injuries']} injuries).")
            mi = m.get("model_info", {})
            st.markdown(f"**No leakage between training and test:** {mi.get('split', 'split by athlete')}. Athletes in "
                        f"both sets: **{mi.get('athletes_in_both_train_and_test', '—')}**, so every test day belongs to "
                        "an athlete the model never saw. The cut-off was set on "
                        f"{mi.get('cut_off_chosen_on', 'training athletes only')}.")
            st.caption("These results are on real runners' data and replace the interim report's figures (AUC 0.71), "
                       "which came from synthetic data in a different evaluation setting; the two are not comparable. "
                       "The model is a prototype trained and tested on runners, not a validated injury predictor "
                       "for ISL footballers.")
            st.caption(f"Model in use: {MODEL_VERSION} · metrics computed {mi.get('trained_at', '—')} with seed "
                       f"{mi.get('seed', '—')}. If the server's scikit-learn version differs from the saved model's, the "
                       "app retrains the same model at start-up (same data, same seed, same cut-off rule).")
            tr = pd.DataFrame([{"Model": k, "AUC": v["auc"], "Flagged": f"{v['flag_rate']:.0%}",
                                "Recall (injuries caught)": f"{v['recall']:.0%}", "Precision": f"{v['precision']:.1%}",
                                "Lift vs base rate": v["lift_vs_base_rate"], "TP": v["tp"], "FP": v["fp"], "FN": v["fn"]}
                               for k, v in m["test_results"].items()])
            st.dataframe(tr, hide_index=True, use_container_width=True)
            st.caption(f"Base rate: {m['base_rate']:.2%} of athlete-days are followed by an injury, so per-day precision "
                       "looks small; the **💰 Cost of errors** tab converts it to weekly squad numbers.")
            st.markdown(DAYS_VS_WEEKS)
            gb_key = next((k for k in m["test_results"] if k.startswith("Gradient boosting")), None)
            if gb_key:
                gb = m["test_results"][gb_key]
                cv = m["cv_5fold_by_athlete"]
                with st.expander("❓ Why logistic regression and not gradient boosting?", expanded=False):
                    st.markdown(
                        f"With the **same cut-off rule** (catch half of the training injuries), gradient boosting caught "
                        f"**{gb['tp']} of {gb['tp'] + gb['fn']}** test injuries while flagging {gb['flag_rate']:.0%} of days; "
                        f"logistic regression caught **{lr['tp']}** while flagging {lr['flag_rate']:.0%} "
                        f"(AUC {gb['auc']:.3f} vs {lr['auc']:.3f}). Across 5 folds the AUCs are "
                        f"{cv['gradient_boosting']['mean_auc']} ± {cv['gradient_boosting']['sd']} vs "
                        f"{cv['logistic_regression']['mean_auc']} ± {cv['logistic_regression']['sd']}, so the gap is about "
                        "one standard deviation: real but small and not stable.\n\n"
                        "We deploy logistic regression because a doctor must sign off every decision: each player's "
                        "**top 3 reasons** come straight from the model's weights and can be checked by hand. Gradient "
                        "boosting would need an extra explanation tool (e.g. SHAP) whose output is harder to verify. "
                        "With real club data we would re-test both and switch if boosting is clearly better.")
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
                card("Flagged / week", f"{flags_week:.1f}", f"{flag_rate:.0%} of weekly player checks")
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
                       "Recall and flag rate come from the held-out test athletes (scored per athlete-day, applied here as "
                       "one check per player per week); the other numbers are the assumptions above.")

    with etabs[2]:
        if not ev:
            st.info("Evaluation results not found: run `python eval/run_evaluation.py`.")
        else:
            def answered_by(o):
                used = o.get("models_used") or {o.get("model", "?"): o.get("attacks", o.get("questions", 0))}
                return ", ".join(f"{k.replace('openai/', '')} ×{v}" for k, v in sorted(used.items(), key=lambda t: -t[1]))

            st.info(f"**Which model?** The live app uses **{MAIN_MODEL}** by default. If Groq rate-limits it, the app "
                    f"switches to **{SMALL_MODEL}** and every briefing shows the model that actually answered. The test "
                    "runs below list the model that answered each one: heavy testing used up the main model's free-tier "
                    "quota, so most test answers came from the small model.")
            a, b_, c = ev["A_retrieval"], ev["B_input_screen"], ev["C_output_checks"]
            st.markdown("**1 · Offline tests** (no AI calls)")
            x1, x2, x3 = st.columns(3)
            with x1:
                card("Retrieval", f"{a['covered_correct'] + a['uncovered_correct']}/{a['covered_total'] + a['uncovered_total']}",
                     f"{a['covered_correct']}/{a['covered_total']} covered · {a['uncovered_correct']}/{a['uncovered_total']} 'not covered'")
            with x2:
                card("Attacks blocked by keyword screen", f"{b_['blocked_v2']}/{b_['attacks_total']}",
                     f"v1 rules {b_['blocked_v1']}/20 · normal notes wrongly blocked {b_['benign_wrongly_blocked']}/10")
            with x3:
                card("Planted bad outputs caught", f"{c['correct']}/{c['cases']}",
                     f"Python output checks (v1 checks: {c.get('correct_v1', c['correct'])}/{c['cases']})")
            if "G_prompt_guard" in ev:
                g = ev["G_prompt_guard"]
                st.markdown("**2 · Prompt Guard 2 classifier** (runs after the keyword screen in the live app)")
                if g.get("unavailable", 0) == len(g.get("rows", [])):
                    st.warning("Prompt Guard 2 did not answer on this key during the test, so only the keyword screen "
                               "was measured.")
                else:
                    g1, g2, g3 = st.columns(3)
                    with g1:
                        card("Attacks flagged", f"{g['attacks_flagged_by_prompt_guard']}/{g['attacks_total']}",
                             f"with keywords: {g['attacks_blocked_combined']}/{g['attacks_total']} blocked")
                    with g2:
                        card("Harder attacks flagged", f"{g['hard_attacks_flagged_by_prompt_guard']}/{g['hard_attacks_total']}",
                             f"with keywords: {g['hard_attacks_blocked_combined']}/{g['hard_attacks_total']} blocked")
                    with g3:
                        card("Normal notes flagged", f"{g['normal_notes_flagged_by_prompt_guard']}/{g['normal_notes_total']}",
                             "false alarms", "#b71c1c" if g["normal_notes_flagged_by_prompt_guard"] else "#1b5e20")
            else:
                st.markdown("**2 · Prompt Guard 2 classifier**")
                st.caption("Not tested yet: run `python eval/run_evaluation.py --guard`. Until then, the input-screen "
                           "numbers above are for the keyword screen only.")
            if "D_llm_end_to_end" in ev:
                d = ev["D_llm_end_to_end"]
                st.markdown(f"**3 · Live AI tests** — 25 questions + 20 attacks · answered by: {answered_by(d)}")
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
            for key_, title in [("E_hard_attacks", "4 · Harder attacks, main run"),
                                ("E_hard_attacks_small", "5 · Harder attacks, small-model run")]:
                if key_ in ev:
                    e = ev[key_]
                    st.markdown(f"**{title}** — {e['attacks']} subtle attacks (Hinglish, contradicting the risk level, "
                                f"misusing clause numbers, treatment advice, comparisons) · requested "
                                f"{e.get('requested_model', e['model']).replace('openai/', '')} · answered by: {answered_by(e)}")
                    z1, z2, z3 = st.columns(3)
                    with z1:
                        card("Blocked by keyword screen", f"{e['blocked_at_input']}/{e['attacks']}", "by design: no trigger words")
                    with z2:
                        card("Caught by output checks", f"{e['llm_answers_flagged_v2']}/{e['attacks']}",
                             "broken format or rule → retry / template", "#b71c1c")
                    with z3:
                        card("Passed every check", f"{e['passed_all_checks_v2']}/{e['attacks']}",
                             "read by hand: did any obey the attack?")
            if "E_hard_attacks" in ev:
                st.warning("**Manual review of the harder attacks (our 28 Oct runs):** no answer obeyed an attack, except one that got past "
                           "every check in both runs: a note claiming *“his sleep was actually 9.5 hours”* made the AI "
                           "write *“C-8 does not apply”*, although his data says 6.5 h. Fix: a new output check rejects any "
                           "briefing that calls a clause triggered by the player's own data “not applicable”, and prompt "
                           "rule 12 says notes never override measured data (planted-output test above). The AI also "
                           "often answers “not covered” when a clause does apply — safe, but unhelpful.")
            st.markdown("**Test rows**")
            if "D_llm_end_to_end" in ev:
                with st.expander("Live AI test rows"):
                    st.dataframe(pd.DataFrame(ev["D_llm_end_to_end"]["rows"]), hide_index=True, use_container_width=True)
            if "E_hard_attacks" in ev:
                with st.expander("Harder attack rows (AI's first answer)"):
                    st.dataframe(pd.DataFrame(ev["E_hard_attacks"]["rows"]), hide_index=True, use_container_width=True)
            if "G_prompt_guard" in ev:
                with st.expander("Prompt Guard 2 rows"):
                    st.dataframe(pd.DataFrame(ev["G_prompt_guard"]["rows"]), hide_index=True, use_container_width=True)
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
        try:
            acc_rows, fm_rows = get_acceptance(PROTOCOL.raw, os.path.getmtime(EVAL_PATH) if os.path.exists(EVAL_PATH) else 0)
        except Exception as e:  # noqa: BLE001
            acc_rows, fm_rows = [], []
            st.error(f"Could not run the acceptance suite: {str(e)[:200]}")
        if acc_rows:
            n_pass = sum(r["Status"].startswith("✅") for r in acc_rows)
            n_fail = sum(r["Status"].startswith("❌") for r in acc_rows)
            n_wait = len(acc_rows) - n_pass - n_fail
            a1, a2, a3 = st.columns(3)
            with a1:
                card("Passed", f"{n_pass}/{len(acc_rows)}", "requirements met", "#1b5e20")
            with a2:
                card("Failed", n_fail, "needs fixing before use", "#b71c1c" if n_fail else "#1b5e20")
            with a3:
                card("Not run here", n_wait, "need the live-AI test run (Groq key)", "#475569")
            st.write("")
            st.caption("One row per requirement of the brief. Rows marked *live check* are re-run by this page from the "
                       "code that serves the app (no AI call); the live-AI rows come from `eval/run_evaluation.py`.")
            st.dataframe(pd.DataFrame(acc_rows), hide_index=True, use_container_width=True, height=38 + 35 * len(acc_rows),
                         column_config={"ID": st.column_config.TextColumn(width="small"),
                                        "Requirement": st.column_config.TextColumn(width="medium"),
                                        "Expected": st.column_config.TextColumn(width="medium"),
                                        "Result": st.column_config.TextColumn(width="large"),
                                        "Evidence": st.column_config.TextColumn(width="medium"),
                                        "Status": st.column_config.TextColumn(width="small")})
            st.download_button("📥 Download acceptance results (CSV)", pd.DataFrame(acc_rows).to_csv(index=False),
                               "acceptance_tests.csv", "text/csv")
        if fm_rows:
            st.markdown(f"**Failure modes** — {sum(r['passed'] for r in fm_rows)}/{len(fm_rows)} handled as expected "
                        "(simulated AI client, no network)")
            st.dataframe(pd.DataFrame([{"Case": r["case"], "Expected": r["expected"], "What happened": r["got"],
                                        "OK": "✅" if r["passed"] else "❌"} for r in fm_rows]),
                         hide_index=True, use_container_width=True,
                         column_config={"What happened": st.column_config.TextColumn(width="large"),
                                        "OK": st.column_config.TextColumn(width="small")})

    with etabs[6]:
        st.markdown("""
- **Population shift:** trained on 74 elite runners, used on footballers. Running distance maps to GPS distance, but football adds contact injuries and match load the model has never seen.
- **Modest accuracy:** test AUC ≈ 0.70; most signal comes from *previous injury*, which alone scores 0.67. The model is a triage aid, not a predictor of individual injuries.
- **Missing data in the source:** 19 of 74 athletes never logged 'training success', so that feature was dropped rather than letting blanks look like low scores.
- **Sleep is not in the model** (no real data); it only triggers protocol clause C-8. The demo squad's sleep values are illustrative.
- **Cut-off drift:** set on training athletes to catch 50% of injuries; on unseen athletes it caught 31% while flagging 16%.
- **Guardrails were tuned on our own test sets**; fresh attacks (e.g. in the peer challenge) may get further. Checks catch broken rules, not subtle changes of meaning — a human still reads every briefing before sign-off.
- **Access control is a prototype:** one shared staff PIN and one sign-off PIN, not individual accounts; the 5-attempt lock is per browser session. A live system needs personal logins.
- **Which model answered the tests:** heavy testing used up the free-tier quota of gpt-oss-120b, so most test answers came from gpt-oss-20b. The live app shows the model that answered every briefing.
- **Over-caution:** the AI often answers "not covered by the protocol" when a clause does apply (e.g. a misquoted C-3). Safe, but less helpful.
- **The AI can be persuaded by notes:** in testing, a note claiming better sleep made it call clause C-8 "not applicable". A new output check now blocks that pattern, but other phrasings may slip through — the medical sign-off is the final safeguard.
- **Storage is a prototype:** sign-offs, data edits and the audit log are kept in one SQLite file shared by everyone using the app, so a decision made on one device shows on any other. Streamlit Cloud wipes that file when the app is rebooted or redeployed. A live version needs a managed database with backups.
- **Privacy:** prototype uses only public, anonymised data. Groq's free tier must not receive real player health data; a live version needs a paid tier with a data-processing agreement or a self-hosted model (DPDP Act 2023).
""")

# ------------------------------------------------------------------ 6. audit log
EVENT_NAMES = {
    "briefing_generated": "📄 Briefing generated", "signoff": "✅ Signed off", "input_blocked": "🛡️ Input blocked",
    "signoff_rejected_pin": "🔒 Wrong sign-off PIN", "entry_saved": "✍️ Data saved", "entry_rejected": "⛔ Data rejected",
    "csv_upload": "📤 CSV uploaded", "demo_reset": "🔄 Demo reset", "staff_pin_rejected": "🔒 Wrong staff PIN",
    "coach_note_blocked": "🛡️ Coach note blocked", "coach_view_opened": "👀 Coach view opened",
}


def summarise_event(event, details):
    try:
        d = json.loads(details or "{}")
    except ValueError:
        return ""
    if event == "briefing_generated":
        s = (f"{d.get('generated_by')}" + (f" ({str(d['model_used']).replace('openai/', '')})" if d.get("model_used") else "")
             + f" · {d.get('attempts')} attempt(s) · clauses {', '.join(d.get('clauses', []))}")
        s += f" · {len(d['guardrail_problems'])} check(s) failed on the way" if d.get("guardrail_problems") else ""
        return s + (f" · assessment {d['assessment_id']}" if d.get("assessment_id") else "")
    if event == "signoff":
        s = f"{d.get('decision')} → {d.get('final_status')} (model said {d.get('model_level')})"
        s += " · against the model, reason given" if d.get("against_model") else ""
        return s + (f" · decision {d['decision_id']}" if d.get("decision_id") else "") + \
            (f" · replaces {d['supersedes']}" if d.get("supersedes") else "")
    if event in ("staff_pin_rejected", "signoff_rejected_pin"):
        return f"attempt {d.get('attempt', '?')} of {MAX_PIN_TRIES}" + (f" as {d['role']}" if d.get("role") else "")
    if event == "coach_note_blocked":
        return "; ".join(d.get("problems", []))[:140]
    if event == "input_blocked":
        return "; ".join(f"{lbl}: {why[0][:60]}" for lbl, why in d.get("blocked", []))
    if event == "entry_rejected":
        return f"{len(d.get('errors', []))} problem(s): " + "; ".join(d.get("errors", []))[:120]
    if event == "entry_saved":
        return f"{d.get('km_this_week')} km this week · exertion {d.get('exertion')}/10 · sleep {d.get('sleep_hours')} h"
    if event == "coach_view_opened":
        return f"{d.get('signed_off_visible', '?')} signed-off player(s) visible"
    if event == "csv_upload":
        return f"{d.get('rows_loaded')} rows loaded, {d.get('rows_rejected')} rejected"
    return ""


with etabs[4]:
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
                     column_config={"Timestamp": st.column_config.TextColumn(width="small"),
                                    "Event": st.column_config.TextColumn(width="small"),
                                    "Player": st.column_config.TextColumn(width="small"),
                                    "By": st.column_config.TextColumn(width="medium"),
                                    "What happened": st.column_config.TextColumn(width="large")})
        st.download_button("📥 Download full audit log (CSV, with details)", adf.to_csv(index=False),
                           "audit_log.csv", "text/csv")
    else:
        st.info("No events yet. Generate a briefing or sign off a player.")

# ------------------------------------------------------------------ 7. protocol
with etabs[5]:
    st.subheader("Club protocol (knowledge base for retrieval)")
    st.caption("Read fresh from knowledge/protocol.md on every run. Changing a threshold in an `Applies when:` line "
               "changes which clauses are retrieved, with no code change.")
    st.dataframe(pd.DataFrame([{"Clause": c.cid, "Title": c.title, "Applies when": c.condition, "Text": c.text}
                               for c in PROTOCOL.clauses]), hide_index=True, use_container_width=True, height=420,
                 column_config={"Text": st.column_config.TextColumn(width="large")})
    with st.expander("Raw protocol.md"):
        st.code(PROTOCOL.raw, language="markdown")

# ------------------------------------------------------------------ open the Player assessment tab after "Open"
# st.tabs cannot be switched from Python, so a tiny script clicks the 2nd top-level tab in the browser,
# then the player's first sub-tab (① Briefing).
# If the browser blocks it, nothing breaks: the player is already selected in the Player assessment tab.
if ss.pop("_goto_pending", False):
    with st.sidebar:
        components.html(
            "<script>/* open #%d */ (function () { var tries = 0; function go() { try {"
            " var d = window.parent.document;"
            " var root = d.querySelector('[data-testid=stMain]') || d.querySelector('section.main') || d;"
            " var lists = root.querySelectorAll('[role=tablist]');"
            " if (lists.length) { var t = lists[0].querySelectorAll('[role=tab]');"
            " if (t.length > 1) { t[1].click(); setTimeout(function () { try {"
            " var l2 = root.querySelectorAll('[role=tablist]');"
            " if (l2.length > 1) { var s2 = l2[1].querySelectorAll('[role=tab]'); if (s2.length) { s2[0].click(); } }"
            " } catch (e2) {} }, 150); return; } } } catch (e) { return; }"
            " if (tries++ < 30) { setTimeout(go, 100); } } go(); })();</script>" % ss.get("_goto_nonce", 0),
            height=0)
