"""Metadata saved with every result: seed, config, git commit, time."""

import datetime
import json
import subprocess
from pathlib import Path


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_json(path, payload, config, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run": {"git_commit": git_commit(), "seed": seed, "config": config,
                       "time": datetime.datetime.now().isoformat(timespec="seconds")},
               **payload}
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
