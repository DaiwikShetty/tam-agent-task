"""Knowledge-base loader. Returns raw articles - retrieval is your job.

The KB is roughly 100k tokens. Do not send the whole KB to the model per ticket.
"""
from pathlib import Path
import re

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def _parse(text: str) -> tuple[dict, str]:
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        meta[k.strip()] = v
    return meta, text[m.end():]


def load_kb() -> list[dict]:
    """Return a list of {"id", "meta", "body"} for every article."""
    out = []
    for p in sorted(KB_DIR.glob("*.md")):
        meta, body = _parse(p.read_text())
        out.append({"id": meta["id"], "meta": meta, "body": body})
    return out
