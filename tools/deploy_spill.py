#!/usr/bin/env python3
"""Install or revert the Strata SSD-tier candidate without overwriting the old engine."""

import json
import os
import shutil
import sys
from pathlib import Path

base = Path("/mnt/datos/Strata")
cfg = base / "strata-swift-iq3_xxs.json"
backup = base / "strata-swift-iq3_xxs.json.pre-spill"
new_bin = base / "engine/strata-spill"
old_bin = base / "engine/strata"
cache = Path("/mnt/datos/Strata-conversation-cache")
key = base / "strata-api-key"

if sys.argv[1:] == ["rollback"]:
    if not backup.exists():
        raise SystemExit("missing original config backup")
    previous = json.loads(backup.read_text())
    previous["api_key"] = key.read_text().strip()
    temporary = cfg.with_name(cfg.name + ".tmp-rollback")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(previous, output, indent=2)
        output.write("\n")
    os.replace(temporary, cfg)
    cfg.chmod(0o600)
    print("Original engine settings restored with LAN authentication. Restart server to apply.")
    raise SystemExit(0)
if sys.argv[1:] != ["install"]:
    raise SystemExit("usage: deploy_spill.py install|rollback")
if backup.exists():
    raise SystemExit("backup already exists; refusing to overwrite it")
if not old_bin.is_file() or new_bin.exists() or not key.is_file():
    raise SystemExit("missing original engine/key or new engine already exists")
original = json.loads(cfg.read_text())
assert original["exe"] == str(old_bin)
assert original["host"] == "0.0.0.0" and original["port"] == 11435
assert original["args"][original["args"].index("--max-context") + 1] == "262000"
assert "--kv-grow" in original["args"]
assert not any(x in original.get("env", {}) for x in ("STRATA_QFUSE", "STRATA_GDN_SPLIT"))
args = ["--conversation-cache-spill-dir", str(cache),
        "--conversation-cache-disk-mib", "32768", "--conversation-cache-ttl-s", "86400",
        "--conversation-cache-disk-min-free-mib", "192", "--conversation-cache-n-min", "64"]
assert not any(x in original["args"] for x in args[::2])
shutil.copy2(cfg, backup)
shutil.copy2("/mnt/datos/Strata-spill/build/strata", new_bin)
cache.mkdir(mode=0o700, exist_ok=True)
cache.chmod(0o700)
modified = dict(original)
modified["exe"] = str(new_bin)
modified["args"] = original["args"] + args
modified["api_key"] = key.read_text().strip()
temporary = cfg.with_name(cfg.name + ".tmp-spill")
fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as output:
    json.dump(modified, output, indent=2)
    output.write("\n")
os.replace(temporary, cfg)
cfg.chmod(0o600)
print("Installed separate engine and private authenticated config; backup:", backup)
print("RAM minimum for streamed SSD operations: 192 MiB; insufficient memory falls back to prompt reread")
