#!/usr/bin/env python3
"""The SSD tier must retain a second checkpoint for a long conversation that branches."""

import json
import re
import urllib.request
from pathlib import Path

root = Path("/mnt/datos/Strata-spill")
log = root / "sparse-integration-engine.log"
url = "http://127.0.0.1:11436/v1/chat/completions"


def ask(messages):
    data = json.dumps({"model": "swift-1.5-iq3_xxs", "messages": messages,
                       "max_tokens": 192, "temperature": 0, "enable_thinking": False}).encode()
    with urllib.request.urlopen(urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}), timeout=240) as response:
        result = json.load(response)
    choice = result["choices"][0]
    assert choice["message"].get("content") and choice["finish_reason"] in ("stop", "length"), choice
    return result


start = log.stat().st_size
base = {"role": "system", "content":
        "Esta es la prueba larga de checkpoints SSD. " +
        "Estás ayudando con un pequeño proyecto de matemáticas en español. " * 3500}
hello = {"role": "user", "content": "hola"}
first = ask([base, hello])
other = ask([{"role": "system", "content": "Eres un ayudante de cocina."},
             {"role": "user", "content": "Escribe una frase sobre cocinar arroz."}])
# A changed assistant answer branches after the shared first turn. The new prompt should hit the older
# checkpoint, instead of rereading the entire (longer) shared prefix.
continued = ask([base, hello, {"role": "assistant", "content": "¡Hola!"},
                 {"role": "user", "content": "Dime un número primo."}])
events = log.read_bytes()[start:].decode(errors="replace")
lines = [line for line in events.splitlines() if "disk session:" in line or
         "conversation cache: restored" in line or "strata serve: prompt " in line]
print("\n".join(lines), flush=True)
assert "retained=2" in events, "no sparse checkpoint was saved"
assert "conversation cache: restored" in events and "(disk/checkpoint)" in events, "no branch restored from SSD"
prompts = re.findall(r"strata serve: prompt (\d+) tokens = (\d+) reused", events)
assert len(prompts) >= 3 and int(prompts[-1][1]) > 0, "branch prompt reread from zero"
print("SSD sparse branch test passed:", [r["usage"]["prompt_tokens"] for r in (first, other, continued)],
      "restored", prompts[-1][1], "tokens", flush=True)
