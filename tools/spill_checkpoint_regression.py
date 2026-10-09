#!/usr/bin/env python3
"""A shared-prefix sibling after 'hola' must not lose its checkpoint to the disk tier."""

import json
import os
import urllib.request
from pathlib import Path

base = "http://127.0.0.1:11436"
log = Path("/mnt/datos/Strata-spill/checkpoint-regression-engine.log")
system = {"role": "system", "content": "Responde en español. Prueba de rama del checkpoint sin perder contexto."}


def ask(messages, limit):
    body = json.dumps({"model": "swift-1.5-iq3_xxs", "messages": messages,
                       "temperature": 0, "max_tokens": limit}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "X-Session-Id": "checkpoint-A",
                                          **({"Authorization": "Bearer " + os.environ["STRATA_TEST_API_KEY"]}
                                             if os.getenv("STRATA_TEST_API_KEY") else {})})
    with urllib.request.urlopen(req, timeout=240) as response:
        data = json.load(response)
    choice = data["choices"][0]
    text = choice["message"].get("content") or ""
    print("finish:", choice["finish_reason"], "prompt:", data["usage"]["prompt_tokens"],
          "answer:", repr(text[:200]), flush=True)
    assert choice["finish_reason"] in ("stop", "length") and text
    return choice["message"]


first = {"role": "user", "content": "hola"}
log_start = log.stat().st_size
ask([system, first], 150)
# Some clients resend only the visible text, rather than the complete assistant turn. The
# prompt then shares the first turn's checkpoint, but not its running-state/live prefix.
reply = {"role": "assistant", "content": "¡Hola!"}
answer = ask([system, first, reply,
              {"role": "user", "content": "dame los primeros 100 numeros primos"}], 3072)
assert "541" in (answer.get("content") or "")
events = log.read_bytes()[log_start:].decode(errors="replace")
assert "disk session deferred (rewinding to checkpoint" in events, events[-4000:]
assert "restoring a conversation checkpoint failed" not in events
print("Sibling rewind answered without losing the checkpoint", flush=True)
