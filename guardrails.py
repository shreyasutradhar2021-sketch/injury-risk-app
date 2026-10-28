"""
Guardrails (security layer).

Input side:  injection / jailbreak screen on free text (staff notes, staff question):
             1) keyword rules, 2) Llama Prompt Guard 2 classifier on Groq (if reachable).
Output side: every LLM briefing is checked in Python before anyone sees it
             (same idea as the professor's "LLM proposes, Python validates").
"""

import re
from typing import Dict, List, Optional, Set, Tuple

MAX_TEXT_LEN = 600
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"

INJECTION_PATTERNS = [
    r"ignore (all |the |any |your )?(previous|prior|above|earlier)?\s*(instructions|rules|protocol|prompt)",
    r"disregard (the |all |your )?(rules|protocol|instructions)",
    r"forget (the |all |your )?(rules|protocol|instructions)",
    r"\b(system|developer) prompt\b",
    r"reveal (your|the) (rules|instructions|prompt)",
    r"\byou are now\b",
    r"\bact as\b",
    r"\bpretend\b",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"(mark|declare|report|say|state)\s+(him|her|them|the player|P-\d+)?\s*(as\s+)?(fit|cleared|available|healthy|injured|unfit)",
    r"\b(clear|clearance)\b.*\b(play|match|selection)\b",
    r"\bbench (him|her|them)\b",
    r"\bnew instructions?\b",
    r"\boverride\b",
    r"other players?'?s? (data|scores|records|names)",
    r"\b(all|every) players?\b.*\b(data|scores|records|list)\b",
    r"</?(clauses|player_data|staff_notes|system)>",
    r"\bbase64\b",
    # v2 additions after the first red-team run (see prompt log):
    r"forget (the |all |your )?(clauses|rules|protocol|instructions)",
    r"\b(say|state|write|report)\s+(that\s+)?(he|she|the player)\s+(is|was)\s+(fit|cleared|injured|unfit)",
    r"\b(list|print|show|reveal|translate|repeat|include)\b.*\b(rules|instructions|prompt)\b",
    r"\b(turn off|disable|bypass|switch off|ignore)\b.*\b(safety|guardrails?|filters?|rules)\b",
    r"\b(make up|invent|create|add)\b.*\bclause\b",
]
INJECTION_PATTERNS_V1_COUNT = 19

BANNED_OUTPUT = [
    (r"\b(is|are|be|been|now)\s+(cleared|fit to play|fit for|unfit|injured)\b", "states clearance/injury status"),
    r"\bcleared to\b",
    (r"\b(bench|drop|exclude|rule out)\s+(him|her|the player|them|from)\b", "makes a selection decision"),
    (r"\bdiagnos(e|is|ed)\b", "suggests a diagnosis"),
    (r"\b(tear|torn|fracture|sprain|ligament damage)\b", "names a specific injury"),
    (r"\b(should|must|will)\s+(not\s+)?play\b", "makes a playing decision"),
    (r"you write reports for the club medical staff", "leaks the system prompt"),
]


def screen_text(text: str, groq_client=None, player_id: str = "", all_player_ids: List[str] = None,
                patterns: List[str] = None) -> Tuple[bool, List[str]]:
    """Return (allowed, reasons). Blocks injection attempts in staff free text."""
    reasons = []
    if not text:
        return True, reasons
    patterns = INJECTION_PATTERNS if patterns is None else patterns
    # Least privilege: free text about one player may not name another player.
    other_ids = [p for p in re.findall(r"\bP-\d+\b", text, flags=re.I) if p.upper() != player_id.upper()]
    if other_ids:
        reasons.append(f"mentions another player ({other_ids[0]})")
    if len(text) > MAX_TEXT_LEN:
        reasons.append(f"text longer than {MAX_TEXT_LEN} characters")
    low = text.lower()
    for pat in patterns:
        if re.search(pat, low if pat != r"\bDAN\b" else text):
            reasons.append(f"matched injection rule: {pat}")
            break
    if groq_client is not None and not reasons:
        verdict = prompt_guard(text, groq_client)
        if verdict is not None and verdict >= 0.5:
            reasons.append(f"Prompt Guard 2 score {verdict:.2f}")
    return (len(reasons) == 0), reasons


def prompt_guard(text: str, client) -> Optional[float]:
    """Llama Prompt Guard 2 on Groq. Returns probability of an attack, or None if unavailable."""
    try:
        resp = client.chat.completions.create(
            model=PROMPT_GUARD_MODEL, messages=[{"role": "user", "content": text}], temperature=0)
        out = (resp.choices[0].message.content or "").strip()
        m = re.search(r"[01](?:\.\d+)?", out)
        if m:
            return float(m.group(0))
        if "malicious" in out.lower() or "injection" in out.lower() or "jailbreak" in out.lower():
            return 1.0
        if "benign" in out.lower():
            return 0.0
    except Exception:
        return None
    return None


_NUM = re.compile(r"(?<![A-Za-z-])(\d+(?:\.\d+)?)")


def numbers_in(text: str) -> List[float]:
    return [float(x) for x in _NUM.findall(text or "")]


def allowed_numbers(rec: Dict, result: Dict, clause_texts: List[str]) -> Set[float]:
    """Numbers the briefing may mention: inputs, derived values, and numbers in retrieved clauses."""
    vals = set()
    for v in rec.values():
        try:
            vals.add(round(float(v), 2))
        except (TypeError, ValueError):
            pass
    f = result["features"]
    vals |= {round(f["acwr"], 2), round(f["acwr"], 1), round(f["hi_share"] * 100), round(f["hi_share"] * 100, 1),
             round(result["risk_index"], 1), round(result["risk_index"]), float(result["percentile"]),
             round(result["probability"] * 100, 1), round(result["probability"] * 100, 2)}
    for t in clause_texts:
        vals |= {round(x, 2) for x in numbers_in(t)}
    vals |= {float(i) for i in range(0, 25)}  # small counts, clause numbers, list numbering
    return vals


def validate_briefing(briefing: Dict, *, rec: Dict, result: Dict, retrieved_ids: List[str],
                      clause_texts: List[str], player_id: str, all_player_ids: List[str],
                      question_covered: Optional[bool]) -> List[str]:
    """Return a list of problems. Empty list = briefing passes."""
    problems = []
    if briefing.get("risk_level") != result["risk_level"]:
        problems.append(f"risk_level must be '{result['risk_level']}' (the model's output), got '{briefing.get('risk_level')}'")

    text_parts = [briefing.get("summary", ""), briefing.get("for_medical_team", ""),
                  briefing.get("answer_to_question", "") or ""]
    text_parts += [a.get("action", "") for a in briefing.get("protocol_actions", [])]
    text_parts += [k for k in briefing.get("key_factors", [])]
    text_parts += briefing.get("conflicts", []) or []
    full = " ".join(text_parts)

    # Cited clauses must have been retrieved
    cited = set(re.findall(r"C-\d+", full)) | {a.get("clause", "") for a in briefing.get("protocol_actions", [])}
    cited.discard("")
    bad = sorted(c for c in cited if c not in retrieved_ids)
    if bad:
        problems.append(f"cites clauses that were not retrieved: {', '.join(bad)}")
    if not briefing.get("protocol_actions"):
        problems.append("no protocol actions with clause citations")

    # Numbers must match the inputs or the retrieved clauses
    ok = allowed_numbers(rec, result, clause_texts)
    stray = sorted({n for n in numbers_in(re.sub(r"C-\d+|P-\d+", "", full))
                    if not any(abs(n - a) <= 0.011 for a in ok)})
    if stray:
        problems.append(f"numbers not found in the inputs or clauses: {', '.join(f'{x:g}' for x in stray[:6])}")

    # Policy: no clearance / benching / diagnosis / prompt leaks
    for item in BANNED_OUTPUT:
        pat, why = (item if isinstance(item, tuple) else (item, "states clearance"))
        if re.search(pat, full, flags=re.I):
            problems.append(f"policy violation: {why}")

    # Least privilege: no other player's ID
    others = [p for p in all_player_ids if p != player_id and re.search(rf"\b{re.escape(p)}\b", full)]
    if others:
        problems.append(f"mentions other players: {', '.join(others[:3])}")

    # Not-covered questions must get the fixed answer
    if question_covered is False:
        ans = (briefing.get("answer_to_question") or "").strip()
        if "not covered by the protocol" not in ans.lower():
            problems.append("question is not covered by the protocol, but the answer did not say so")
    return problems
