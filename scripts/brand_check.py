"""Fail if any blocklisted real organisation name appears in tracked files.

The blocklist is stored only as SHA-256 hashes of lower-cased 1-3 word phrases,
so the names themselves never appear in the repo."""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

BLOCK = set(Path(__file__).with_name("brand_blocklist.sha256").read_text().split())
files = subprocess.run(["git", "ls-files"], check=False, capture_output=True, text=True).stdout.split()
hits = 0
for f in files:
    try:
        words = re.findall(r"[a-z0-9]+", Path(f).read_text(errors="ignore").lower())
    except (IsADirectoryError, FileNotFoundError):
        continue
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            if hashlib.sha256(" ".join(words[i : i + n]).encode()).hexdigest() in BLOCK:
                print(f"blocklisted name in {f}")
                hits += 1
print(f"brand check: {hits} hits across {len(files)} files")
sys.exit(1 if hits else 0)
