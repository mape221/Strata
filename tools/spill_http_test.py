#!/usr/bin/env python3
"""HTTP chat API: two independent conversations switching through the automatic SSD tier."""

import json
import os
import re
import urllib.request
from pathlib import Path

URL = os.environ.get("STRATA_TEST_URL", "http://127.0.0.1:11436/v1/chat/completions")
ROOT = Path("/mnt/datos/Strata-spill")


def request(messages, max_tokens=384, session_id=None):
    body = {"model": "swift-1.5-iq3_xxs", "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if session_id:
        headers["X-Session-Id"] = session_id
    if os.getenv("STRATA_TEST_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["STRATA_TEST_API_KEY"]
    query = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                   headers=headers)
    with urllib.request.urlopen(query, timeout=360) as response:
        data = json.load(response)
    choice = data["choices"][0]
    if choice.get("finish_reason") not in ("stop", "length"):
        raise AssertionError(f"unexpected finish: {choice.get('finish_reason')}")
    return choice["message"], data.get("usage", {})


def main():
    system_a = {"role": "system", "content": (
        "Responde en español. En esta conversación analizamos matemáticas elementales. " * 65)}
    system_b = {"role": "system", "content": (
        "Responde en español. Esta conversación trata de literatura y océanos. " * 70)}
    a = [system_a, {"role": "user", "content": "dame los primeros 100 numeros primos"}]
    b = [system_b, {"role": "user", "content": "Escribe un poema breve sobre el mar."}]
    a1, au1 = request(a, 3072, "http-integration-A")
    a.append(a1)
    numbers = [(int(i), int(n)) for i, n in re.findall(
        r"(?m)^\s*(\d+)\.\s+(\d+)\s*$", a1.get("content") or "")]
    expected = [n for n in range(2, 542)
                if all(n % d for d in range(2, int(n ** 0.5) + 1))]
    print("A1 prime list:", len(numbers), "last:", numbers[-1] if numbers else None,
          "valid:", [n for _, n in numbers] == expected, flush=True)
    b1, bu1 = request(b, session_id="http-integration-B")
    b.append(b1)
    a.append({"role": "user", "content": "¿Cuál es el primo número 100 de la lista que acabas de dar?"})
    a2, au2 = request(a, session_id="http-integration-A")
    b.append({"role": "user", "content": "Resume tu poema en una frase."})
    b2, bu2 = request(b, session_id="http-integration-B")
    report = {"A1": {"message": a1, "usage": au1},
              "B1": {"message": b1, "usage": bu1},
              "A2": {"message": a2, "usage": au2},
              "B2": {"message": b2, "usage": bu2}}
    (ROOT / "http-test-results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("prompt tokens A1/B1/A2/B2:", [r["usage"].get("prompt_tokens") for r in report.values()], flush=True)
    print("A2 answer:", repr((a2.get("content") or "")[:280]), flush=True)
    print("B2 answer:", repr((b2.get("content") or "")[:220]), flush=True)
    assert [n for _, n in numbers] == expected, "100-prime quality regression"
    assert "541" in (a2.get("content") or ""), "A did not retain the context of the first question"


if __name__ == "__main__":
    main()
