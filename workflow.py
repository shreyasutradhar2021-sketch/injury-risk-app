"""
Human-checkpoint rules, kept apart from the Streamlit page so they can be tested directly
(acceptance suite in the app's Evaluation tab, and eval/run_evaluation.py Part F).
"""

from typing import Any, Dict, List, Optional, Tuple

STATUS_OPTIONS = [
    "Train as planned",
    "Modified load (medical plan)",
    "Medical assessment before next session",
]
SIGNOFF_ROLES = ("Club doctor", "Head physiotherapist")
# What the coach can be told (protocol C-16): fixed messages, no medical details. An optional extra line
# is allowed but must pass guardrails.screen_coach_note (no body parts, symptoms, treatment or numbers).
COACH_MESSAGES = [
    "Available for selection decisions; full training.",
    "Available; training load adjusted this week by the medical team.",
    "Not available until the medical team's review is complete.",
    "Medical team review before the next session; update to follow.",
]
MIN_REASON = 10


def decision_state(signoff: Optional[Dict[str, Any]], current_hash: str) -> str:
    """'none' = never signed; 'stale' = signed on different weekly data (superseded, hidden from the coach);
    'done' = signed on exactly this week's data (the only state the coach can see)."""
    if not signoff:
        return "none"
    return "done" if signoff.get("data_hash") == current_hash else "stale"


def signoff_problem(role: str, pin_ok: bool, confirmed: bool, final_status: Optional[str],
                    coach_msg: Optional[str], decision: str, reason: str, needs_caution: bool,
                    note_problems: List[str]) -> Optional[Tuple[str, str]]:
    """First reason the sign-off must be refused, as (code, message), or None if it can be saved."""
    reason = (reason or "").strip()
    if role not in SIGNOFF_ROLES:
        return "role", "Only the club doctor or head physiotherapist can sign off."
    if not pin_ok:
        return "pin", "Wrong or missing sign-off PIN. Nothing was signed."
    if final_status not in STATUS_OPTIONS or coach_msg not in COACH_MESSAGES:
        return "missing", "Choose a training status and a message for the coach."
    if decision in ("Modify", "Override") and len(reason) < MIN_REASON:
        return "reason", f"Please give a reason (at least {MIN_REASON} characters) for a Modify or Override."
    if needs_caution and final_status == "Train as planned" and len(reason) < MIN_REASON:
        return "reason", ("The model or protocol flags this player (High risk, conflict or data check). "
                          f"Choosing 'Train as planned' needs a reason of at least {MIN_REASON} characters.")
    if note_problems:
        return "note", ("The extra line for the coach was not saved: it " + "; ".join(note_problems) +
                        ". Medical details stay in the staff-only reason box.")
    if not confirmed:
        return "confirm", "Tick the box to confirm you reviewed the briefing and that this decision is yours."
    return None
