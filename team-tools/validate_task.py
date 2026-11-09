#!/usr/bin/env python3
"""Checks a task folder against the rules in AGENTS.md. usage: team-tools/validate_task.py tasks/<name> [--strict-media]"""
import json, os, re, sys
from pathlib import Path

REQUIRED = ["title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores", "media"]
CATEGORIES = {"Impact on jobs", "Impact on AI timelines", "Safety and alignment", "Societal relevance", "Other"}
d = Path(sys.argv[1])
strict_media = "--strict-media" in sys.argv
errs, warns = [], []
spec_p = d / "task.json"
if not spec_p.is_file() or not (d / "README.md").is_file():
    sys.exit("task.json and README.md are required")
spec = json.loads(spec_p.read_text())
for k in REQUIRED:
    if k == "media" and not strict_media:
        continue
    if not str(spec.get(k, "")).strip():
        errs.append(f"task.json: '{k}' missing/empty")
if spec.get("category") not in CATEGORIES:
    errs.append(f"category must be one of {sorted(CATEGORIES)}")
if strict_media and not re.fullmatch(r"M-[A-Z0-9]{6}", str(spec.get("media", ""))):
    errs.append("media ID missing: run tools/upload_media.py first")
# blind review: task text must not identify the team; secrets must never be committed
text = " ".join(str(v) for v in spec.values())
for pat in filter(None, os.environ.get("BLIND_TERMS", "").split(",")):
    if pat.lower() in text.lower():
        errs.append(f"task.json mentions team-identifying term '{pat}'")
tok = os.environ.get("ROBOCURVE_TOKEN")
for f in d.rglob("*"):
    if f.is_file() and f.suffix in {".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}:
        s = f.read_text(errors="ignore")
        if tok and tok in s:
            errs.append(f"{f}: contains the team token")
        if re.search(r"/home/\w+|/Users/\w+", s):
            warns.append(f"{f}: has a user home path (fine in code, not in task text/videos)")
    if f.is_file() and f.suffix in {".mp4", ".mov"}:
        errs.append(f"{f}: videos go to the task board, not the repo")
for w in warns: print("WARN", w)
for e in errs: print("ERROR", e)
sys.exit(1 if errs else 0)
