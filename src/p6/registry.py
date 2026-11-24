"""Run log for the analysis scripts."""
from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "results" / "registry.jsonl"


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def environment() -> dict:
    import numpy, scipy, sklearn
    import os
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "machine": platform.processor(),
            "device": os.environ.get("P6_DEVICE", "cpu")}


class Run:

    def __init__(self, experiment: str, config: dict, outputs: list[str] | None = None):
        self.rec = {"run_id": uuid.uuid4().hex[:12], "experiment": experiment, "config": config,
                    "code_sha": code_hash(), "env": environment(), "outputs": outputs or []}

    def __enter__(self):
        self.rec["start"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.rec["seconds"] = round(time.time() - self._t0, 1)
        self.rec["status"] = "completed" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        with REGISTRY.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.rec) + "\n")
        return False
