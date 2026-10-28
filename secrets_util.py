"""
Find the Groq API key without ever printing it. Order:
1. Streamlit secrets (`groq_api_key` in .streamlit/secrets.toml or the Cloud secrets box)
2. Environment variable GROQ_API_KEY
3. credentials.json with {"groq_api_key": "..."}  (same format as the professor's sample; local only, never commit)
"""

import json
import os
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_groq_key() -> Optional[str]:
    try:
        import streamlit as st  # noqa: WPS433
        key = st.secrets.get("groq_api_key") or st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key).strip()
    except Exception:  # noqa: BLE001  (no streamlit / no secrets file)
        pass
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"].strip()
    path = os.path.join(ROOT, "credentials.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                key = json.load(f).get("groq_api_key", "").strip()
            if key and key != "gsk_...":
                return key
        except Exception:  # noqa: BLE001
            return None
    return None


def get_groq_client():
    key = get_groq_key()
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:  # noqa: BLE001
        return None
