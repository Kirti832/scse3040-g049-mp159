import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")

for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
