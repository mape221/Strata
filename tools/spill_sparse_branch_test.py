#!/usr/bin/env python3
"""A rewrite before the deepest saved checkpoint should restore an earlier sparse checkpoint."""

import json
import re
import urllib.request
from pathlib import Path

log = Path("/mnt/datos/Strata-spill/sparse-integration-engine.log")
url = "http://127.0.0.1:11436/v1/chat/completions"


def ask(messages, session_id):
    body = json.dumps({"model": "swift-1.5-iq3_xxs", "messages": messages,
                       "max_tokens": 192, "temperature": 0, "enable_thinking": False}).encode()
    request = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json", "X-Session-Id": session_id})
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)
    assert result["choices"][0]["message"].get("content")
    return result["usage"]["prompt_tokens"]


start = log.stat().st_size
ask([{"role": "system", "content": "Eres un ayudante de cocina."},
     {"role": "user", "content": "Escribe una frase sobre cocinar arroz."}], "sparse-B")
phrase = "Estás ayudando con un pequeño proyecto de matemáticas en español. "
system = "Esta es la prueba larga de checkpoints SSD. " + phrase * 1000 + "Cambio de rama. " + phrase * 2500
length = ask([{"role": "system", "content": system}, {"role": "user", "content": "hola"},
              {"role": "assistant", "content": "¡Hola!"},
              {"role": "user", "content": "Dime un número primo."}], "sparse-A")
events = log.read_bytes()[start:].decode(errors="replace")
print("\n".join(x for x in events.splitlines() if "conversation cache: restored" in x or
                "strata serve: prompt " in x or "disk session skipped" in x), flush=True)
checkpoints = re.findall(r"conversation cache: restored (\d+) tokens \(disk/checkpoint\)", events)
assert checkpoints and 0 < int(checkpoints[-1]) < 20000, "old SSD checkpoint not used"
print("Old branch restored", checkpoints[-1], "of", length, "prompt tokens", flush=True)
