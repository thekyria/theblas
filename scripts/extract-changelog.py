#!/usr/bin/env python3
"""Print a versioned section from the committed CHANGELOG.md.

Usage: extract-changelog.py <VERSION>

Fails if the requested release section is missing or empty.
"""
import re
import sys

if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
    sys.exit("Usage: extract-changelog.py <VERSION> (X.Y.Z)")

version = sys.argv[1]

with open("CHANGELOG.md", encoding="utf-8") as f:
    content = f.read()

m = re.search(
    rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]:|\Z)",
    content,
    re.MULTILINE | re.DOTALL,
)
if not m:
    sys.exit(f"No changelog section for {version}")

notes = m.group(1).strip()
if not notes:
    sys.exit(f"Empty changelog section for {version}")
print(notes)
