#!/usr/bin/env python3
"""Real-model A/B/A/B + restart test for automatic on-disk conversation switching."""

import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path("/mnt/datos/Strata-spill")
sys.path.insert(0, str(ROOT / "tools"))
from batch_test import Engine, tokenizer  # noqa: E402


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generate(engine, ids, count=48):
    engine.send(f"GEN {count} " + ",".join(map(str, ids)))
    tokens = []
    for line in engine.lines():
        if line.startswith("T "):
            tokens.append(int(line.split()[1]))
        elif line.startswith("DONE "):
            fields = line.split()
            return tokens, int(fields[8]) if len(fields) > 8 else 0
        elif line.startswith("ERR "):
            raise RuntimeError(f"engine request failed: {line}")
    raise RuntimeError("engine ended without DONE")


def follow(tok, prompt, answer, question):
    end = tok.encode("<|im_end|>\n", parse_special=True)
    return prompt + answer + ([] if answer and answer[-1] in end else end) + tok.encode(chat(question), parse_special=True)


def run():
    cfg = json.loads(Path("/mnt/datos/Strata/strata-swift-iq3_xxs.json").read_text())
    tok = tokenizer(cfg["tokenizer"])
    para = ("The history of the printing press begins with movable type in East Asia and continues with Gutenberg's "
            "press in Mainz around 1450, which spread across Europe within decades and lowered the cost of books. ")
    repeat = int(os.getenv("SPILL_REPEAT", "38"))
    pa = tok.encode(chat(para * repeat + "\nSummarize this document."), parse_special=True)
    pb = tok.encode(chat("Write a short poem about the sea and explain its metaphors. " * 20), parse_special=True)
    name = os.getenv("SPILL_CACHE_NAME", "cache-test-3")
    cache_dir = ROOT / name
    extra = ["--conversation-cache-spill-dir", str(cache_dir),
             "--conversation-cache-disk-mib", "32768", "--conversation-cache-ttl-s", "86400",
             "--conversation-cache-disk-min-free-mib", "192", "--conversation-cache-n-min", "64",
             "--session-min-free-mib", "4096", "--pcie-frac", "0", "--adapt-every", "1000000"]
    os.environ["BATCH_TEST_LOG"] = str(ROOT / f"integration-{name}.log")

    def start():
        return Engine(str(ROOT / "build" / "strata"), cfg, 0, {"STRATA_IQ_MT_MIN": "1"}, extra)

    engine = start()
    stop = threading.Event()
    minimum_available = [1 << 60]

    def sample_ram():
        while not stop.wait(0.2):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    minimum_available[0] = min(minimum_available[0], int(line.split()[1]) * 1024)
                    break

    watcher = threading.Thread(target=sample_ram, daemon=True)
    watcher.start()
    try:
        a1, _ = generate(engine, pa)
        b1, _ = generate(engine, pb)
        pa2 = follow(tok, pa, a1, "Give two related keywords.")
        a2, reused_a = generate(engine, pa2)
        pb2 = follow(tok, pb, b1, "Give a third metaphor.")
        b2, reused_b = generate(engine, pb2)
        engine.send("QUIT")
        engine.p.wait(timeout=180)
        log = Path(os.environ["BATCH_TEST_LOG"]).read_text()
        (ROOT / f"integration-{name}-first.log").write_text(log)
        print("A first prompt tokens:", len(pa), "minimum MemAvailable MiB:",
              minimum_available[0] // (1 << 20), flush=True)
        assert log.count("disk session saved") >= 2, "sessions did not reach disk"
        assert log.count("restored ") >= 2 and "(disk/" in log, "disk restore not observed"
        assert reused_a > 64 and reused_b > 64, (reused_a, reused_b)
        assert len(list(cache_dir.glob("*.sess"))) <= 3, "old versions accumulated after replacing a chat"
        print("A/B/A/B: restored from disk; reused prompt tokens", reused_a, reused_b, flush=True)
    finally:
        stop.set()
        watcher.join(timeout=2)
        if engine.p.poll() is None:
            engine.p.terminate()
            engine.p.wait(timeout=30)

    # The active session at QUIT and all previously switched sessions must survive a restart.
    restarted = start()
    try:
        pa3 = follow(tok, pa2, a2, "Now give one more keyword.")
        _, reused_restart = generate(restarted, pa3)
        restarted.send("QUIT")
        restarted.p.wait(timeout=180)
        log = Path(os.environ["BATCH_TEST_LOG"]).read_text()
        (ROOT / f"integration-{name}-restart.log").write_text(log)
        assert "spill dir ready (" in log and "(disk/" in log, "session not restored after restart"
        assert reused_restart > 64, reused_restart
        print("after restart: restored A from disk; reused", reused_restart, "prompt tokens", flush=True)
    finally:
        if restarted.p.poll() is None:
            restarted.p.terminate()
            restarted.p.wait(timeout=30)

    if os.getenv("SPILL_VERIFY_REFERENCE") == "1":
        # With static expert placement and fixed CPU rounding, compare exactly the same follow-up when A
        # stayed live and when it was saved and restored from disk between the two turns.
        original_log = os.environ["BATCH_TEST_LOG"]
        os.environ["BATCH_TEST_LOG"] = str(ROOT / f"integration-{name}-reference.log")
        ordinary = ["--pcie-frac", "0", "--adapt-every", "1000000"]
        reference = Engine(str(ROOT / "build" / "strata"), cfg, 0, {"STRATA_IQ_MT_MIN": "1"}, ordinary)
        try:
            _, _ = generate(reference, pa)
            a2_reference, reused_reference = generate(reference, pa2)
            print("A follow-up token IDs equal live reference:", a2_reference == a2,
                  "reused reference:", reused_reference, flush=True)
            assert a2_reference == a2, "restoring A changed its continuation"
            if os.getenv("SPILL_PRIMES") == "1":
                ids = tok.encode(chat("dame los primeros 100 numeros primos"), parse_special=True)
                prime_ids, _ = generate(reference, ids, 2048)
                output = tok.decode(prime_ids)
                (ROOT / f"integration-{name}-primes.txt").write_text(output)
                content = output.split("</think>")[-1]
                pairs = [(int(i), int(n)) for i, n in re.findall(r"(?m)^\s*(\d+)\.\s+(\d+)\s*$", content)]
                nums = [n for _, n in pairs]
                expected = [n for n in range(2, 542)
                            if all(n % d for d in range(2, int(n ** 0.5) + 1))]
                print("100 primes valid:", nums == expected,
                      "rows:", len(pairs), "last:", nums[-1] if nums else None, flush=True)
                assert nums == expected, "real quality prompt failed"
            reference.send("QUIT")
            reference.p.wait(timeout=180)
        finally:
            os.environ["BATCH_TEST_LOG"] = original_log
            if reference.p.poll() is None:
                reference.p.terminate()
                reference.p.wait(timeout=30)


if __name__ == "__main__":
    run()
