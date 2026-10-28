"""
Audit log in SQLite: every score, briefing, guardrail event and sign-off decision.
On Streamlit Community Cloud the file resets when the app restarts (fine for a prototype;
a live version would use a managed database).
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

DB_PATH = os.path.join(os.path.dirname(__file__), "audit_log.sqlite")


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, event TEXT, player_id TEXT, actor TEXT, details TEXT)""")
    return con


def log_event(event: str, player_id: str = "", actor: str = "system", details: Dict[str, Any] = None):
    with _conn() as con:
        con.execute("INSERT INTO events (ts, event, player_id, actor, details) VALUES (?,?,?,?,?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, player_id, actor,
                     json.dumps(details or {}, default=str)))


def read_events(limit: int = 500) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT ts, event, player_id, actor, details FROM events ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
    return [{"Timestamp": r[0], "Event": r[1], "Player": r[2], "By": r[3], "Details": r[4]} for r in rows]


def clear_events():
    with _conn() as con:
        con.execute("DELETE FROM events")
