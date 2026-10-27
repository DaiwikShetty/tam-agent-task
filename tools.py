"""Tools the triage agent can call. Every tool returns a small JSON-able dict.

Design rules:
  * Indexes are built once (Tools.__init__), never per ticket.
  * Tools return compact, pre-computed facts (days to renewal, usage deltas), not raw tables.
  * Free text coming from tickets or articles is untrusted and is truncated and labelled as such.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

from .data import AS_OF_DATE
from .search import BM25, error_codes, jaccard, tokenize

AS_OF = pd.Timestamp(AS_OF_DATE) + pd.Timedelta(hours=23, minutes=59)

CANONICAL_MODULES = ["Platform", "Contracts", "Sourcing", "Spend Analytics", "Supplier Management",
                     "eProcurement", "eInvoice", "Integrations"]
_ALIASES = {
    "platform": "Platform", "sso": "Platform", "login": "Platform",
    "contract": "Contracts", "clm": "Contracts", "contract mgmt": "Contracts",
    "sourcing events": "Sourcing", "rfx": "Sourcing", "esourcing": "Sourcing",
    "spend": "Spend Analytics", "analytics": "Spend Analytics", "reports": "Spend Analytics",
    "sim": "Supplier Management", "supplier portal": "Supplier Management", "supplier mgmt": "Supplier Management",
    "eproc": "eProcurement", "e-procurement": "eProcurement", "procurement": "eProcurement",
    "einv": "eInvoice", "e-invoice": "eInvoice", "invoicing": "eInvoice", "ap": "eInvoice",
    "erp integration": "Integrations", "api": "Integrations", "connector": "Integrations",
}
_CANON_LOWER = {m.lower(): m for m in CANONICAL_MODULES}


def normalize_module(raw: str | None) -> str | None:
    r = (raw or "").strip().lower()
    if not r:
        return None
    return _CANON_LOWER.get(r) or _ALIASES.get(r)


def _aslist(v) -> list[str]:
    if isinstance(v, list):
        return v
    return [x.strip() for x in str(v or "").split(",") if x.strip()]


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


_COMPANY_SUFFIX = re.compile(r"\b(inc|llc|ltd|gmbh|corp|corporation|co|plc|sa|ag)\b\.?", re.I)


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", _COMPANY_SUFFIX.sub("", re.sub(r"[^\w\s]", " ", (s or "").lower()))).strip()


class Tools:
    def __init__(self, accounts: pd.DataFrame, tickets: pd.DataFrame, usage: pd.DataFrame, kb: list[dict]):
        self.accounts = accounts.set_index("account_id", drop=False)
        self.by_domain = {d.lower(): a for a, d in zip(accounts["account_id"], accounts["domain"])}
        self.names = {a: _norm_name(n) for a, n in zip(accounts["account_id"], accounts["company_name"])}
        self.usage = usage.sort_values(["account_id", "month"])

        # --- tickets: resolve account for every historical ticket (account_id is blank for ~3.5%) ---
        t = tickets.copy()
        t["acct"] = [a or self.by_domain.get(d.lower(), "") for a, d in zip(t["account_id"], t["requester_domain"])]
        t["ts"] = pd.to_datetime(t["created_at"], errors="coerce")
        text = (t["subject"] + " " + t["description"]).tolist()
        t["tokens"] = [set(tokenize(x)) for x in text]
        t["codes"] = [error_codes(x) | ({e.upper()} if e else set()) for x, e in zip(text, t["error_code"])]
        self.tk = t.set_index("ticket_id", drop=False)

        # --- KB ---
        self.kb = {a["id"]: a for a in kb}
        self.kb_ids = [a["id"] for a in kb]
        docs = []
        for a in kb:
            m = a["meta"]
            head = f"{m.get('title', '')} " * 3 + f"{m.get('module', '')} {' '.join(_aslist(m.get('error_codes')))} "
            docs.append(tokenize(head + a["body"]))
        self.kb_index = BM25(docs)
        self.current: dict | None = None

    # ------------------------------------------------------------------ dispatch
    def bind(self, ticket: dict):
        self.current = ticket

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool '{name}'", "available": self.names_available()}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except KeyError as e:
            return {"error": f"not found: {e}"}

    @staticmethod
    def names_available():
        return ["resolve_account", "get_account_profile", "get_usage_trend", "search_tickets",
                "list_policies", "search_kb", "read_kb_article", "submit_triage"]

    def risk_inputs(self, account_id: str) -> dict:
        """Facts behind the account-risk override in KB-2914 (the model still applies the policy)."""
        a = self.accounts.loc[account_id]
        days = (pd.Timestamp(a["renewal_date"]) - pd.Timestamp(AS_OF_DATE)).days
        u = self.usage[self.usage.account_id == account_id].tail(12)["active_users"].tolist()
        decline = round(100 * (u[-1] - u[-4]) / u[-4], 1) if len(u) >= 4 and u[-4] else None
        esc = int(a["open_escalations"] or 0)
        return {"renewal_within_90_days": 0 <= days <= 90, "days_to_renewal": days,
                "active_users_change_pct_over_last_3_months": decline,
                "users_declined_more_than_25pct": decline is not None and decline < -25,
                "open_escalations": esc, "two_or_more_open_escalations": esc >= 2,
                "all_override_conditions_met": (0 <= days <= 90) and ((decline is not None and decline < -25) or esc >= 2)}

    # ------------------------------------------------------------------ account tools
    def t_resolve_account(self, domain: str = "", company_name: str = "") -> dict:
        """Map requester domain / company name to an account. Domain is authoritative; names are fuzzy."""
        out: dict = {}
        cur = self.current or {}
        if cur.get("account_id"):
            out["ticket_account_id_field"] = cur["account_id"]
        d = (domain or "").strip().lower()
        if d:
            acc = self.by_domain.get(d)
            if acc:
                out |= {"match": acc, "company_name": self.accounts.loc[acc, "company_name"],
                        "method": "domain_exact", "confidence": "high"}
                f = out.get("ticket_account_id_field")
                if f and f != acc:
                    out["warning"] = f"ticket account_id field ({f}) disagrees with domain match ({acc}); trust the domain"
                return out
            out["note"] = f"domain '{d}' matches no account"
        n = _norm_name(company_name)
        if n:
            cands = sorted(((difflib.SequenceMatcher(None, n, v).ratio(), a) for a, v in self.names.items()), reverse=True)[:3]
            out["candidates"] = [{"account_id": a, "company_name": self.accounts.loc[a, "company_name"], "name_similarity": round(s, 2)}
                                 for s, a in cands]
            if len(cands) > 1 and cands[0][0] - cands[1][0] < 0.1:
                out["warning"] = "top candidates are close; similar company names exist. Do not guess; use domain or ask"
            out["method"] = "fuzzy_name"
            out["confidence"] = "high" if cands[0][0] > 0.95 and len(cands) > 1 and cands[0][0] - cands[1][0] >= 0.1 else "low"
        if not out:
            out["error"] = "provide domain or company_name"
        return out

    def t_get_account_profile(self, account_id: str) -> dict:
        a = self.accounts.loc[account_id]
        renewal = pd.Timestamp(a["renewal_date"])
        days = (renewal - pd.Timestamp(AS_OF_DATE)).days
        mods = [m for m in a["modules_licensed"].split(";") if m]
        out = {
            "account_id": account_id, "company_name": a["company_name"], "tier": a["tier"], "region": a["region"],
            "industry": a["industry"], "arr_band": a["arr_band"], "renewal_date": a["renewal_date"],
            "days_to_renewal": days, "renewal_within_90_days": 0 <= days <= 90,
            "open_executive_escalations": int(a["open_escalations"] or 0),
            "last_qbr_date": a["last_qbr_date"] or None, "nps_last": a["nps_last"] or None,
            "modules_licensed": mods,
        }
        if a["sso_enabled_date"]:
            ago = (pd.Timestamp(AS_OF_DATE) - pd.Timestamp(a["sso_enabled_date"])).days
            out["sso_enabled_or_reconfigured"] = a["sso_enabled_date"]
            out["sso_change_days_ago"] = ago
            out["sso_changed_within_30_days"] = ago <= 30
        else:
            out["sso_enabled_or_reconfigured"] = None
        cur = self.current or {}
        mod = normalize_module(cur.get("module"))
        if mod:
            out["ticket_module_normalized"] = mod
            out["ticket_module_licensed"] = mod in mods or mod == "Platform"
        out["risk_override_inputs_per_KB-2914"] = self.risk_inputs(account_id)
        hist = self.tk[(self.tk.acct == account_id) & (self.tk.ts <= (pd.Timestamp(cur["created_at"]) if cur else AS_OF))]
        if cur:
            hist = hist[hist.index != cur["ticket_id"]]
        open_now = hist[hist.status.isin(["open", "pending_customer"])]
        out["other_open_tickets"] = len(open_now)
        out["tickets_last_90d"] = int((hist.ts >= AS_OF - pd.Timedelta(days=90)).sum())
        return out

    def t_get_usage_trend(self, account_id: str) -> dict:
        u = self.usage[self.usage.account_id == account_id].tail(12)
        if u.empty:
            return {"error": f"no usage rows for {account_id}"}

        def chg(col):
            last3, prev3 = u[col].tail(3).mean(), u[col].iloc[-6:-3].mean()
            return round(100 * (last3 - prev3) / prev3, 1) if prev3 else None

        au = u["active_users"].tolist()
        chg3m = round(100 * (au[-1] - au[-4]) / au[-4], 1) if len(au) >= 4 and au[-4] else None
        peak = u["active_users"].max()
        cur_users = int(u["active_users"].iloc[-1])
        out = {
            "account_id": account_id,
            "active_users_change_pct_over_last_3_months": chg3m,
            "active_users_change_pct_last3_vs_prev3": chg("active_users"),
            "documents_change_pct_last3_vs_prev3": chg("documents_processed"),
            "login_rate_change_pct_last3_vs_prev3": chg("login_rate_pct"),
            "active_users_now": cur_users, "active_users_12m_peak": int(peak),
            "drop_from_peak_pct": round(100 * (peak - cur_users) / peak, 1) if peak else None,
            "last_6_months": [f"{r.month}: users={r.active_users} docs={r.documents_processed} login%={r.login_rate_pct} tix={r.tickets_opened}"
                              for r in u.tail(6).itertuples()],
            "note": "documents_processed and login_rate_pct are noisy month to month; active_users is the steady signal",
        }
        d = out["active_users_change_pct_last3_vs_prev3"]
        out["verdict"] = "declining" if (d is not None and d <= -15) or (out["drop_from_peak_pct"] or 0) >= 25 else \
            "growing" if d is not None and d >= 15 else "stable"
        return out

    # ------------------------------------------------------------------ ticket history
    def t_search_tickets(self, account_id: str = "", error_code: str = "", query: str = "", days: int = 90,
                         limit: int = 6) -> dict:
        """Earlier tickets (never later than the current one). Filter by account and/or error code; rank by text similarity."""
        cur = self.current or {}
        cur_ts = pd.Timestamp(cur["created_at"]) if cur.get("created_at") else AS_OF
        days = max(1, min(int(days), 365))
        limit = max(1, min(int(limit), 10))
        base = self.tk[(self.tk.ts <= cur_ts) & (self.tk.ts >= cur_ts - pd.Timedelta(days=days))]
        if cur:
            base = base[base.index != cur["ticket_id"]]
        code = (error_code or "").strip().upper()
        qtok = set(tokenize(query or f"{cur.get('subject', '')} {cur.get('description', '')}"))
        cur_text = f"{cur.get('subject', '')} {cur.get('description', '')}"
        cur_codes = error_codes(cur_text) | ({cur["error_code"].upper()} if cur.get("error_code") else set())
        cur_tok = set(tokenize(cur_text))
        cur_mod = normalize_module(cur.get("module"))
        cur_acct = self.by_domain.get((cur.get("requester_domain") or "").lower()) or cur.get("account_id")
        df = base
        if account_id:
            df = df[df.acct == account_id]
        if code:
            df = df[df.codes.apply(lambda s: code in s)]
        if not account_id and not code and not query:
            return {"error": "give account_id and/or error_code and/or query"}
        rows = []
        for tid, r in df.iterrows():
            sim = jaccard(qtok, r.tokens)
            if code or sim >= 0.12:
                age = (cur_ts - r.ts).days
                r_mod = normalize_module(r.module)
                sim_cur = jaccard(cur_tok, r.tokens)
                # KB-5930: same account, open/pending, same module, same error code or symptom, within previous 7 days
                dup = bool(cur_acct and r.acct == cur_acct and r.status in ("open", "pending_customer") and age <= 7
                           and (not cur_mod or not r_mod or cur_mod == r_mod)
                           and (bool(cur_codes & r.codes) or sim_cur >= 0.3))
                rows.append({
                    "ticket_id": tid, "account_id": r.acct or None, "age_days": age, "status": r.status, "module": r.module or None,
                    "error_codes": sorted(r.codes), "similarity": round(sim, 2),
                    "meets_duplicate_policy_KB-5930": dup,
                    "subject": _clip(r.subject, 90), "description_untrusted": _clip(r.description, 140),
                })
        rows.sort(key=lambda x: (x["similarity"], -x["age_days"]), reverse=True)
        out = {"scope": {"account_id": account_id or None, "error_code": code or None, "days": days},
               "matches_found": len(rows), "results": rows[:limit]}
        if code:  # widespread-incident signal across all accounts
            allc = self.tk[(self.tk.ts <= cur_ts) & self.tk.codes.apply(lambda s: code in s)]
            if cur:
                allc = allc[allc.index != cur["ticket_id"]]
            for label, d in (("last_48h", 2), ("last_30d", 30)):
                w = allc[allc.ts >= cur_ts - pd.Timedelta(days=d)]
                out[f"same_error_code_{label}"] = {"tickets": len(w), "distinct_accounts": int(w.acct.nunique())}
        return out

    def policy_duplicates(self, ticket: dict) -> list[dict]:
        """Earlier tickets that meet the KB-5930 duplicate rule for this ticket (used to cross-check the model)."""
        self.bind(ticket)
        acct = self.by_domain.get((ticket.get("requester_domain") or "").lower()) or ticket.get("account_id")
        if not acct:
            return []
        res = self.t_search_tickets(account_id=acct, days=7, limit=10)
        return [r for r in res.get("results", []) if r.get("meets_duplicate_policy_KB-5930")]

    # ------------------------------------------------------------------ knowledge base
    def _kb_row(self, a: dict, snippet: str = "") -> dict:
        m = a["meta"]
        r = {"id": a["id"], "title": m.get("title"), "type": m.get("type"), "module": m.get("module"),
             "status": m.get("status"), "error_codes": _aslist(m.get("error_codes")), "last_updated": m.get("last_updated")}
        if m.get("superseded_by"):
            r["superseded_by"] = m["superseded_by"]
        if snippet:
            r["snippet"] = snippet
        return r

    def _snippet(self, body: str, q: set[str]) -> str:
        best, best_s = "", -1
        for line in re.split(r"\n+", body):
            if line.lstrip().startswith("#") or line.lstrip().startswith("|---"):
                continue  # headings/table rules only repeat the title
            line = line.strip(" -*>\t")
            if len(line) < 20:
                continue
            s = len(q & set(tokenize(line)))
            if s > best_s:
                best, best_s = line, s
        return _clip(best, 220)

    def t_list_policies(self) -> dict:
        pol = [self._kb_row(a) for a in self.kb.values() if a["meta"].get("type") == "policy"]
        return {"count": len(pol), "policies": pol[:60],
                "note": "policy articles define priority, routing, escalation, TAM-notification and duplicate rules; read the relevant ones"}

    def t_search_kb(self, query: str, type: str = "", module: str = "", error_code: str = "", k: int = 5) -> dict:
        k = max(1, min(int(k), 8))
        qtoks = tokenize(f"{query} {error_code}")
        codes = error_codes(f"{query} {error_code}")
        scores = self.kb_index.scores(qtoks)
        mod = normalize_module(module) or (module or "").strip()
        ranked = []
        for i, s in scores.items():
            a = self.kb[self.kb_ids[i]]
            m = a["meta"]
            if type and m.get("type") != type:
                continue
            if mod and str(m.get("module", "")).lower() not in (mod.lower(), "", "all", "platform"):
                continue
            if codes & {c.upper() for c in _aslist(m.get("error_codes"))}:
                s += 8.0  # exact error-code hit
            if m.get("status") == "deprecated":
                s *= 0.5
            ranked.append((s, a))
        ranked.sort(key=lambda x: -x[0])
        q = set(qtoks)
        res = [self._kb_row(a, self._snippet(a["body"], q)) | {"score": round(s, 1)} for s, a in ranked[:k]]
        out = {"query": query, "results": res}
        if any(r["status"] == "deprecated" for r in res):
            out["warning"] = "some results are deprecated; read them for the superseded_by pointer, but cite the current article"
        if not res:
            out["note"] = "no hits; try different keywords, drop filters, or search by error code"
        return out

    def t_read_kb_article(self, article_id: str) -> dict:
        a = self.kb.get(article_id)
        if not a:
            return {"error": f"no such article '{article_id}'", "hint": "ids look like KB-0000; use ids from search results"}
        m = a["meta"]
        body = a["body"].strip()
        out = self._kb_row(a) | {"body": body[:4000] + ("\n[truncated]" if len(body) > 4000 else "")}
        rel = []
        for rid in _aslist(m.get("related")):
            r = self.kb.get(rid)
            rel.append({"id": rid, "title": r["meta"].get("title"), "status": r["meta"].get("status")} if r else {"id": rid, "missing": True})
        out["related"] = rel
        if m.get("status") == "deprecated":
            out["warning"] = f"DEPRECATED. Superseded by {m.get('superseded_by') or 'unknown'}; read and cite that one instead"
        if m.get("status") == "active_incident":
            out["warning"] = "this article describes an ACTIVE INCIDENT; check whether the ticket matches its symptoms"
        return out
