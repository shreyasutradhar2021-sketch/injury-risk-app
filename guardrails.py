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


def prompt_guard_raw(text: str, client) -> Tuple[Optional[float], str]:
    """Llama Prompt Guard 2 on Groq: (probability of an attack or None, raw reply or error)."""
    try:
        resp = client.chat.completions.create(
            model=PROMPT_GUARD_MODEL, messages=[{"role": "user", "content": text}], temperature=0)
        out = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        return None, f"error: {str(e)[:120]}"
    m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", out)
    if m:
        return float(m.group(1)), out
    low = out.lower()
    if "malicious" in low or "injection" in low or "jailbreak" in low or "unsafe" in low:
        return 1.0, out
    if "benign" in low or "safe" in low:
        return 0.0, out
    return None, out


def prompt_guard(text: str, client) -> Optional[float]:
    """Probability that the text is an attack, or None if the classifier is unavailable."""
    return prompt_guard_raw(text, client)[0]


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


# v2 output checks, added after a code review of weak spots (see PROMPT_LOG #11):
# Hinglish/Hindi status phrases, health-status claims, treatment advice, load increases
# no clause allows, comparisons with other players, and a summary that contradicts the risk level.
BANNED_OUTPUT_V2 = [
    (r"\bfit\s+(hai|he|h)\b|\bkhel\s*(sakta|sakti|lega|legi|ega)\b|\bkhelega\b|\bbench\s+(karo|kar)\b",
     "states fitness or selection in Hindi/Hinglish"),
    (r"\b(has|had)\s+no\s+injur\w*|\bis\s+(fine|healthy|ok|okay|fully fit)\b", "states the player's health status"),
    (r"\bplay(s|ing)?\s+(the\s+)?(full|whole|entire)\b|\b90\s+minutes\b|\bstart(s)?\s+(on|the)\s+(saturday|sunday|match|game)",
     "makes a playing decision"),
    (r"\b(mri|x-ray|ultrasound|scan|anti-inflammator\w*|ibuprofen|paracetamol|painkiller\w*|medication|\d+\s*mg)\b",
     "gives treatment or medication advice"),
    (r"\b(than|compared (to|with)|like)\s+(the\s+)?(other|rest)\b", "compares with other players"),
]


def _v2_checks(briefing: Dict, result: Dict, retrieved_ids: List[str], rule_ids: Optional[List[str]] = None) -> List[str]:
    problems = []
    free_text = " ".join([briefing.get("summary", ""), briefing.get("for_medical_team", ""),
                          briefing.get("answer_to_question", "") or ""] + list(briefing.get("key_factors", [])))
    everything = free_text + " " + " ".join(a.get("action", "") for a in briefing.get("protocol_actions", []))
    if re.search(r"[\u0900-\u097F]", everything):
        problems.append("reply is not in English")
    for pat, why in BANNED_OUTPUT_V2:
        if re.search(pat, everything, flags=re.I):
            problems.append(f"policy violation: {why}")
    # The summary must not contradict the model's risk level.
    level = result["risk_level"].lower()
    for m in re.finditer(r"(\bnot\s+)?\b(low|high)[\s-]+risk\b|risk\s+(?:level\s+)?(?:is|was|as|of)\s+(low|high)\b",
                         free_text, flags=re.I):
        said = (m.group(2) or m.group(3) or "").lower()
        if said and said != level and not m.group(1):
            problems.append(f"text says '{said} risk' but the model's risk level is {result['risk_level']}")
            break
    # A clause matched by the player's own numbers cannot be declared "not applicable" (found in testing:
    # a staff note claimed 9.5 h sleep and the AI wrote "C-8 does not apply", although the data said 6.5 h).
    data_clauses = set(rule_ids if rule_ids is not None else retrieved_ids)
    for m in re.finditer(r"\b(C-\d+)\b[^.]{0,40}?\b(does not|doesn't|no longer|not)\s+appl(y|ies)\b", everything, flags=re.I):
        if m.group(1) in data_clauses:
            problems.append(f"says {m.group(1)} does not apply, but the player's own data triggered it")
            break
    # Load increases: only clause C-7 (under-loading) allows increases, and only 10-15%.
    for m in re.finditer(r"\b(increas\w*|rais\w*|go(?:es)?\s+up)\b[^.]{0,60}?\bby\s+(\d+)\s*(%|percent)",
                         everything, flags=re.I):
        pct = float(m.group(2))
        if pct > 15 or "C-7" not in retrieved_ids:
            problems.append(f"suggests increasing load by {pct:g}%, which no retrieved clause allows")
            break
    return problems


def validate_briefing(briefing: Dict, *, rec: Dict, result: Dict, retrieved_ids: List[str],
                      clause_texts: List[str], player_id: str, all_player_ids: List[str],
                      question_covered: Optional[bool], version: int = 2,
                      rule_ids: Optional[List[str]] = None,
                      conflict_pairs: Optional[List[Tuple[str, str]]] = None) -> List[str]:
    """Return a list of problems. Empty list = briefing passes. version=1 runs the original checks only."""
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
    if version >= 2:
        problems += _v2_checks(briefing, result, retrieved_ids, rule_ids)
        # Conflicting clauses must be named in "conflicts" and resolved with C-17 (prompt rule 8).
        conflict_text = " ".join(briefing.get("conflicts", []) or [])
        for a, b in conflict_pairs or []:
            if not (re.search(rf"\b{re.escape(a)}\b", conflict_text) and re.search(rf"\b{re.escape(b)}\b", conflict_text)):
                problems.append(f"does not state the conflict between {a} and {b}")
        if conflict_pairs and "C-17" not in cited:
            problems.append("clauses conflict but C-17 (the more cautious option applies) is not cited")
    return problems



# ---------------------------------------------------------------- coach note (human-written)
# Protocol C-16: the coach sees only the signed-off status and a short logistics note - no diagnoses,
# body parts, symptoms, treatment or wellness numbers. Found in user testing: "Hamstring strain
# suspected, sleep 6.5 h" reached the coach word for word. The note is now built from fixed choices
# plus an optional extra line that must pass this screen.
MEDICAL_TERMS = [
    r"injur\w*", r"strain\w*", r"sprain\w*", r"tear\w*", r"torn", r"fractur\w*", r"broken", r"rupture\w*",
    r"concuss\w*", r"diagnos\w*", r"suspect\w*", r"symptom\w*", r"pain\w*", r"sore\w*", r"ach(e|es|ing)", r"swell\w*",
    r"inflam\w*", r"tight\w*", r"cramp\w*", r"illness", r"ill", r"sick", r"fever", r"infection", r"virus", r"covid",
    r"hamstring\w*", r"quad\w*", r"calf", r"calves", r"groin", r"knee\w*", r"ankle\w*", r"hip\w*", r"thigh\w*", r"achilles",
    r"acl", r"mcl", r"menisc\w*", r"ligament\w*", r"tendon\w*", r"muscle\w*", r"shoulder\w*", r"foot", r"feet",
    r"neck", r"wrist", r"hamstrings", r"adductor\w*", r"doms",
    r"mri", r"scan\w*", r"x-?ray\w*", r"ultrasound", r"medicat\w*", r"tablet\w*", r"pill\w*", r"ibuprofen", r"paracetamol",
    r"painkiller\w*", r"anti-?inflammator\w*", r"inject\w*", r"physio(therapy)? (treatment|session)s?", r"rehab\w*",
    r"sleep\w*", r"slept", r"insomnia", r"exertion", r"rpe", r"fatigue\w*", r"tired\w*", r"exhaust\w*", r"wellness",
    r"acwr", r"workload", r"risk", r"score\w*", r"heart rate", r"hrv", r"mood", r"stress\w*", r"anxi\w*", r"depress\w*",
    r"mental", r"bmi", r"pregnan\w*", r"hospital\w*", r"surgery",
]
_MED_RE = re.compile(r"\b(" + "|".join(MEDICAL_TERMS) + r")\b", re.I)
_TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s?(am|pm)\b", re.I)


def screen_coach_note(text: str, player_id: str = "", all_player_ids: List[str] = None) -> List[str]:
    """Return problems with a free-text line that will be shown to the head coach (empty = OK)."""
    problems = []
    text = (text or "").strip()
    if not text:
        return problems
    if len(text) > 160:
        problems.append("keep it under 160 characters")
    found = sorted({m.group(0).lower() for m in _MED_RE.finditer(text)})
    if found:
        problems.append("contains medical or wellness details (" + ", ".join(found[:4]) + ")")
    if re.search(r"\d", _TIME_RE.sub("", text)):
        problems.append("contains numbers (only times such as 10:30 or 4 pm are allowed)")
    others = [p for p in re.findall(r"\bP-\d+\b", text, flags=re.I) if p.upper() != (player_id or "").upper()]
    if others:
        problems.append(f"mentions another player ({others[0]})")
    ok, why = screen_text(text, None, player_id=player_id, all_player_ids=all_player_ids)
    if not ok and not others:
        problems.append("looks like an instruction rather than a note")
    return problems
