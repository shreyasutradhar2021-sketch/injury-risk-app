"""
Shared store in SQLite:
- events:   the audit log (every briefing, blocked input, data change, PIN failure and sign-off)
- signoffs: the current medical decision per player, shared by ALL browser sessions, so a decision
            made on the doctor's laptop is what the coach sees on any other device.
- entries:  weekly data entered in the app (one row per player), shared the same way.

On Streamlit Community Cloud the file lives on the app's own disk: it is shared by everyone using the
app, survives normal use, but is wiped when the app is rebooted or redeployed (disclosed in the app).
A live version would use a managed database with per-user accounts.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

DB_PATH = os.path.join(os.path.dirname(__file__), "audit_log.sqlite")
IST = timezone(timedelta(hours=5, minutes=30))  # the club is in India; servers run on UTC


def now_ist(fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.now(IST).strftime(fmt) + " IST"


def _conn():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, event TEXT, player_id TEXT, actor TEXT, details TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS signoffs (
        player_id TEXT PRIMARY KEY, record TEXT, updated TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS entries (
        player_id TEXT PRIMARY KEY, record TEXT, updated TEXT, by TEXT)""")
    return con


def log_event(event: str, player_id: str = "", actor: str = "system", details: Dict[str, Any] = None):
    with _conn() as con:
        con.execute("INSERT INTO events (ts, event, player_id, actor, details) VALUES (?,?,?,?,?)",
                    (now_ist("%Y-%m-%d %H:%M:%S"), event, player_id, actor,
                     json.dumps(details or {}, default=str)))


def read_events(limit: int = 500) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT ts, event, player_id, actor, details FROM events ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
    return [{"Timestamp": r[0], "Event": r[1], "Player": r[2], "By": r[3], "Details": r[4]} for r in rows]


def clear_events():
    with _conn() as con:
        con.execute("DELETE FROM events")


# ---------------------------------------------------------------- shared sign-offs
def save_signoff(player_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Save (or replace) the current decision for a player. Returns the stored record with its decision_id."""
    record = dict(record)
    record.setdefault("decision_id", uuid.uuid4().hex[:8])
    record["player_id"] = player_id
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO signoffs (player_id, record, updated) VALUES (?,?,?)",
                    (player_id, json.dumps(record, default=str), now_ist("%Y-%m-%d %H:%M:%S")))
    return record


def load_signoffs() -> Dict[str, Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT player_id, record FROM signoffs").fetchall()
    out = {}
    for pid, rec in rows:
        try:
            out[pid] = json.loads(rec)
        except ValueError:
            continue
    return out


def clear_signoffs():
    with _conn() as con:
        con.execute("DELETE FROM signoffs")


# ---------------------------------------------------------------- shared weekly data entries
def _plain(v):
    """numpy numbers -> plain Python numbers, so they come back as numbers (not strings) from JSON."""
    return v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes)) else v


def save_entry(player_id: str, record: Dict[str, Any], by: str = ""):
    """Weekly data entered in the app (overrides the demo row for that player, for every viewer)."""
    record = {k: _plain(v) for k, v in record.items() if not str(k).startswith("_")}
    with _conn() as con:
        con.execute("INSERT OR REPLACE INTO entries (player_id, record, updated, by) VALUES (?,?,?,?)",
                    (player_id, json.dumps(record, default=str), now_ist(), by))


def load_entries() -> Dict[str, Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT player_id, record, updated, by FROM entries").fetchall()
    out = {}
    for pid, rec, updated, by in rows:
        try:
            r = json.loads(rec)
        except ValueError:
            continue
        r["_entered"] = f"entered by {by} at {updated}"
        out[pid] = r
    return out


def clear_entries():
    with _conn() as con:
        con.execute("DELETE FROM entries")
