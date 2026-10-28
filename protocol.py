"""
Protocol knowledge base + retrieval (the "R" in RAG).

Two kinds of retrieval:
1. Rule-based: each clause has an `Applies when:` line (e.g. `acwr > 1.5`). The
   player's numbers are checked against it, so the right clauses are always found.
2. Text search (TF-IDF): for a free-text question from staff. If no clause is
   similar enough, the answer is "Not covered by the protocol".

The file is read fresh on every call (same idea as procurement-rules.md in the
professor's sample), so editing protocol.md changes behaviour without code changes.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "protocol.md")
NOT_COVERED = "Not covered by the protocol: refer to the medical team."


@dataclass
class Clause:
    cid: str
    title: str
    condition: str
    text: str

    def as_block(self) -> str:
        return f"[{self.cid}] {self.title}: {self.text}"


@dataclass
class Protocol:
    clauses: List[Clause]
    settings: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    @property
    def by_id(self) -> Dict[str, Clause]:
        return {c.cid: c for c in self.clauses}

    def setting(self, key: str, default: float) -> float:
        try:
            return float(self.settings.get(key, default))
        except ValueError:
            return default

    def conflict_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for part in self.settings.get("conflict_pairs", "").split(","):
            if ":" in part:
                a, b = [p.strip() for p in part.split(":", 1)]
                pairs.append((a, b))
        return pairs


def load_protocol(path: str = PROTOCOL_PATH) -> Protocol:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing protocol file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    settings = {}
    m = re.search(r"^Settings:\s*\n((?:- .*\n?)+)", raw, flags=re.M)
    if m:
        for line in m.group(1).splitlines():
            if "=" in line:
                k, v = line.lstrip("- ").split("=", 1)
                settings[k.strip()] = v.strip()

    clauses = []
    for block in re.split(r"^## ", raw, flags=re.M)[1:]:
        lines = block.strip().splitlines()
        head = re.match(r"(C-\d+)\s+(.*)", lines[0].strip())
        if not head:
            continue
        cond, body = "always", []
        for ln in lines[1:]:
            if ln.lower().startswith("applies when:"):
                cond = ln.split(":", 1)[1].strip()
            elif ln.strip():
                body.append(ln.strip())
        clauses.append(Clause(head.group(1), head.group(2).strip(), cond, " ".join(body)))
    return Protocol(clauses=clauses, settings=settings, raw=raw)


# ---------- rule-based retrieval ----------
_COND = re.compile(r"^\s*([a-z_0-9]+)\s*(==|>=|<=|>|<)\s*([A-Za-z0-9_.]+)\s*$")


def _check_one(expr: str, facts: Dict) -> bool:
    m = _COND.match(expr)
    if not m:
        return False
    key, op, val = m.groups()
    if key not in facts:
        return False
    left = facts[key]
    try:
        right = float(val)
        left = float(left)
    except (TypeError, ValueError):
        left, right = str(left).lower(), str(val).lower()
        return op == "==" and left == right
    return {"==": left == right, ">=": left >= right, "<=": left <= right,
            ">": left > right, "<": left < right}[op]


def condition_holds(cond: str, facts: Dict) -> bool:
    """Evaluate e.g. 'acwr >= 1.3 and acwr <= 1.5' without using eval()."""
    cond = cond.strip()
    if cond == "always":
        return True
    if cond == "question_only":
        return False
    return any(all(_check_one(p, facts) for p in alt.split(" and "))
               for alt in cond.split(" or "))


def retrieve_by_rules(protocol: Protocol, facts: Dict) -> List[Clause]:
    hits = [c for c in protocol.clauses if c.cid != "C-17" and condition_holds(c.condition, facts)]
    ids = {c.cid for c in hits}
    conflicts = [(a, b) for a, b in protocol.conflict_pairs() if a in ids and b in ids]
    if conflicts:
        facts["conflict"] = "yes"
        facts["conflict_pairs"] = conflicts
        c17 = protocol.by_id.get("C-17")
        if c17:
            hits.append(c17)
    else:
        facts["conflict"] = "no"
        facts["conflict_pairs"] = []
    return hits


# ---------- text retrieval for staff questions ----------
def retrieve_by_question(protocol: Protocol, question: str, top_k: int = 2) -> Tuple[List[Clause], List[float]]:
    question = (question or "").strip()
    if not question:
        return [], []
    docs = [f"{c.title}. {c.title}. {c.text}" for c in protocol.clauses]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(docs + [question])
    sims = cosine_similarity(mat[-1], mat[:-1])[0]
    order = sims.argsort()[::-1][:top_k]
    min_sim = protocol.setting("min_similarity", 0.10)
    keep = [(protocol.clauses[i], float(sims[i])) for i in order if sims[i] >= min_sim]
    return [k[0] for k in keep], [k[1] for k in keep]


def retrieve(protocol: Protocol, facts: Dict, question: Optional[str] = None):
    """Return (clauses, info). info says which were found by rules / by the question."""
    rule_hits = retrieve_by_rules(protocol, facts)
    q_hits, q_sims = retrieve_by_question(protocol, question) if question else ([], [])
    # A question that names an existing clause (e.g. "as per C-3 ...") retrieves that clause, so the
    # briefing can correct a misquote instead of answering "not covered". Unknown IDs (C-25) do nothing.
    for cid in re.findall(r"\bC-\d+\b", question or ""):
        c = protocol.by_id.get(cid)
        if c and c.cid not in {h.cid for h in q_hits}:
            q_hits.append(c)
            q_sims.append(1.0)
    seen, merged = set(), []
    for c in rule_hits + q_hits:
        if c.cid not in seen:
            merged.append(c)
            seen.add(c.cid)
    info = {
        "rule_clauses": [c.cid for c in rule_hits],
        "question_clauses": [c.cid for c in q_hits],
        "question_similarities": [round(s, 3) for s in q_sims],
        "question_covered": bool(q_hits) if question else None,
        "conflict_pairs": facts.get("conflict_pairs", []),
    }
    return merged, info
