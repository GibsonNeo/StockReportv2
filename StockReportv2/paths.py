from __future__ import annotations
from pathlib import Path

# Repo root is the folder this file lives in (root-level code layout)
ROOT = Path(__file__).resolve().parent

DATA_DIR    = ROOT / "data"
AUDIT_DIR   = ROOT / "audit"
REPORTS_DIR = ROOT / "reports"

for p in (DATA_DIR, AUDIT_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)
