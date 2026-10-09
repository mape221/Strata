#!/usr/bin/env python3
"""Live HTTP test: ID-gated SSD, separate identities, anonymous requests and restart persistence."""

import json
import os
import re
import urllib.request
from pathlib import Path

root = Path("/mnt/datos/Strata-spill")
log = Path(os.getenv("SESSION_ID_TEST_LOG", root / "session-id-test-engine.log"))
cache = Path(os.getenv("SESSION_ID_TEST_CACHE", root / "cache-test-session-id"))
endpoint = "http://127.0.0.1:11436/v1/chat/completions"


def ask(messages, session_id=None, transport="header"):
    request_body = {"model": "swift-1.5-iq3_xxs", "messages": messages,
                    "temperature": 0, "max_tokens": 128, "enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if session_id is not None:
        if transport == "body":
            request_body["session_id"] = session_id
        else:
            headers["x-session-affinity" if transport == "affinity" else "X-Session-Id"] = session_id
    body = json.dumps(request_body).encode()
    req = urllib.request.Request(endpoint, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    msg = result["choices"][0]["message"]
    assert msg.get("content"), result["choices"]
    return msg


def events(since):
    return log.read_bytes()[since:].decode(errors="replace")


def reused(s):
    match = re.findall(r"strata serve: prompt \d+ tokens = (\d+) reused", s)
    assert match, s[-2400:]
    return int(match[-1])


system_a = {"role": "system", "content": "Responde en español; este es el proyecto A de matemáticas. " * 50}
system_b = {"role": "system", "content": "Responde en español; este es el proyecto B de poesía. " * 50}
user_a = {"role": "user", "content": "hola"}
user_b = {"role": "user", "content": "hola"}
if not list(cache.glob("*.meta")):
    # No ID: requests may use the active prompt cache, but never write/restore an SSD session.
    anonymous = ask([system_a, user_a])
    at_anon = log.stat().st_size
    ask([system_a, user_a, anonymous, {"role": "user", "content": "di 7"}])
    assert reused(events(at_anon)) > 0, "anonymous request lost normal live prompt reuse"
    ask([system_b, user_b])
    assert not list(cache.glob("*.meta")), "anonymous request reached SSD"

at = log.stat().st_size
reply_a = ask([system_a, user_a], "A")
ask([system_b, user_b], "B", "affinity")
assert len(list(cache.glob("*.meta"))) >= 1, "A was not saved after an ID switch"
at_restore = log.stat().st_size
ask([system_a, user_a, reply_a, {"role": "user", "content": "di 7"}], "A")
report = events(at_restore)
assert "conversation cache: restored" in report and "disk/" in report and reused(report) > 0, report[-2400:]

# An identical prompt with another ID must NOT search A's index entry.
ask([system_b, user_b], "B", "affinity")
at_other = log.stat().st_size
ask([system_a, user_a, reply_a, {"role": "user", "content": "di 7"}], "C", "body")
report = events(at_other)
assert "session ID is not indexed" in report and reused(report) == 0, report[-2400:]

# Anonymous calls must never search C or A even if their prompt matches perfectly.
at_anonymous = log.stat().st_size
ask([system_a, user_a, reply_a, {"role": "user", "content": "di 7"}])
report = events(at_anonymous)
assert "disk session miss" not in report and "conversation cache: restored" not in report and reused(report) == 0
assert len(list(cache.glob("*.meta"))) >= 2, "identified sessions were not indexed"
print("HTTP A/B/A, unrelated C and anonymous requests passed; SSD files:",
      len(list(cache.glob("*.meta"))), flush=True)
