"""Data loaders. Loading only - no joins, matching or business logic.

Resolving accounts, interpreting usage and finding related tickets is part of
the task. Note the realistic noise described in data/README.md.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load_accounts() -> pd.DataFrame:
    return pd.read_csv(DATA / "accounts.csv", dtype=str, keep_default_na=False)


def load_tickets() -> pd.DataFrame:
    return pd.read_csv(DATA / "tickets.csv", dtype=str, keep_default_na=False)


def load_usage() -> pd.DataFrame:
    df = pd.read_csv(DATA / "usage_metrics.csv", dtype={"account_id": str, "month": str})
    return df


def load_review_set() -> list[str]:
    return [l.strip() for l in (DATA / "review_set.txt").read_text().splitlines() if l.strip()]


AS_OF_DATE = "2026-10-20"  # treat this as "today" for all date calculations
