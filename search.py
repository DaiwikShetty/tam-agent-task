"""Tiny dependency-free retrieval helpers: tokenizer, BM25, Jaccard."""
import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
ERROR_CODE = re.compile(r"\bE-\d{4}\b", re.I)
STOP = set("""a an the and or of to in on for with is are was were be been it its this that these those we our you your
i my me at by from as not no but if so do does did can could would should will please hi hello team thanks thank
have has had when what how there their they them than then also just any all some very into out up""".split())


_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ities", "ity", "ally", "ers", "er", "ed", "es", "al", "ly", "e", "s")


def stem(w: str) -> str:
    """Very light suffix stripper (approve/approval/approvals/approving -> approv). Leaves codes/numbers alone."""
    if any(c.isdigit() for c in w) or len(w) < 5:
        return w
    for _ in range(2):
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        else:
            break
    return w


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall((text or "").lower()):
        if t in STOP:
            continue
        if re.fullmatch(r"e-\d{4}", t):
            out.append(t)  # error codes stay whole
            continue
        out.append(stem(t))
        # index hyphen parts too (e.g. "e-signature" -> "e", "signature")
        if "-" in t:
            out.extend(stem(p) for p in t.split("-") if len(p) > 1 and p not in STOP)
    return out


def error_codes(text: str) -> set[str]:
    return {m.upper() for m in ERROR_CODE.findall(text or "")}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.dl = [len(d) for d in docs]
        self.avg = (sum(self.dl) / self.n) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.post[term].append((i, tf))

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(query):
            plist = self.post.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avg or 1))
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
