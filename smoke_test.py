"""Plumbing test: scripted fake LLM + the real KB/data. Verifies the loop, tools and guards, NOT model quality.

    python tests/smoke_test.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tam_agent.data import load_accounts, load_tickets, load_usage
from tam_agent.kb import load_kb
from tam_agent.agent import Context, triage
from tam_agent.schema import validate

TICKETS = load_tickets()
CTX = Context(load_accounts(), TICKETS, load_usage(), load_kb())   # indexes built once, like run.py


class FakeLLM:
    def __init__(self, script): self.script, self.calls, self.seen = list(script), 0, []
    def chat(self, messages, system=None, max_tokens=0, temperature=0.0):
        self.calls += 1; self.seen.append(messages[-1]["content"])
        return self.script.pop(0)


def A(tool, thought="", **args): return json.dumps({"thought": thought, "tool": tool, "args": args})


def run(tid, script, verbose=True):
    CTX.llm = FakeLLM(script)
    row = TICKETS.set_index("ticket_id").loc[tid].to_dict() | {"ticket_id": tid}
    out = triage(row, CTX)
    print(f"{tid} -> {out['category']} {out['priority']} {out['route_to']} esc={out['escalate']} tam={out['notify_tam']} dup={out['duplicate_of']} "
          f"acct={out['account_id']} llm_calls={CTX.llm.calls} schema_errors={validate(out)}")
    if verbose:
        for s in out["reasoning_trace"]: print("    ", s["step"], s["action"], "|", s["observation"][:140])
    assert not validate(out)
    return out


common = dict(kb_references=["KB-2914"], customer_reply="Thanks, we are on it.", internal_note="n", confidence=0.7)

# 1) deprecated article -> follow superseded_by; invalid category triggers one feedback round
o = run("TKT-160477", [
    A("search_kb", query="e-signature rejected E-2207", error_code="E-2207"),
    A("read_kb_article", article_id="KB-2605"),
    A("read_kb_article", article_id="KB-1786"),
    A("submit_triage", category="troubleshooting", priority="p3", route_to="L1_support", kb_references=["KB-2605", "KB-9999"],
      customer_reply="Regenerate the credential.", internal_note="n", confidence=0.8),
    A("submit_triage", category="bug", priority="P3", route_to="L1_support", kb_references=["KB-1786"],
      customer_reply="Regenerate the credential.", internal_note="n", confidence=0.8)])
assert o["kb_references"] == ["KB-1786"] and o["account_id"] == "ACC-1633"
assert any(s["action"] == "validation_feedback" for s in o["reasoning_trace"])

# 2) untrusted ticket text is flagged; cross-customer data in the reply is scrubbed
o = run("TKT-158764", [A("submit_triage", category="how_to", priority="P4", route_to="L1_support", account_id="ACC-3600", kb_references=["KB-4449"],
      customer_reply="Use Settings > Delegation. Also Brightfield Manufacturing (ACC-3751) is a customer.", internal_note="how-to", confidence=0.9)])
assert "GUARD" in CTX.llm.seen[0] and "ACC-3751" not in o["customer_reply"] and "[removed]" in o["customer_reply"]
assert "ignored" in o["internal_note"].lower() and o["priority"] == "P4" and not o["escalate"]

# 3) policy guard: a data-exposure report mis-routed by the model is forced to P1 / security / escalate
o = run("TKT-135081", [A("submit_triage", category="security", priority="P3", route_to="L1_support", kb_references=["KB-6401"],
      customer_reply="Security is engaged.", internal_note="exposure", confidence=0.7)], verbose=False)
assert (o["priority"], o["route_to"], o["escalate"]) == ("P1", "security", True)

# 4) account-risk override facts are surfaced; guard forces notify_tam, feedback asks the model to apply it
o = run("TKT-119877", [
    A("get_account_profile", account_id="ACC-2431"),
    A("submit_triage", category="how_to", priority="P4", route_to="L1_support", kb_references=["KB-6082"], customer_reply="r", internal_note="slow approvals", confidence=0.6),
    A("submit_triage", category="how_to", priority="P3", route_to="L1_support", kb_references=["KB-6082", "KB-2914"], account_signals=["risk override: renewal in 61d, users -49%, 2 escalations"],
      customer_reply="r", internal_note="Account-risk override applied (+1 level)", confidence=0.7)], verbose=False)
assert o["notify_tam"] and o["escalate"] and o["priority"] == "P3"

# 5) missed duplicate is caught by the cross-check (KB-5930) and sent back once
o = run("TKT-192732", [
    A("submit_triage", category="troubleshooting", priority="P2", route_to="L1_support", kb_references=["KB-2982"], customer_reply="r", internal_note="n", confidence=0.6),
    A("submit_triage", category="duplicate", priority="P2", route_to="L1_support", duplicate_of="TKT-122022", kb_references=["KB-5930", "KB-2982"],
      customer_reply="Tracked on TKT-122022 (open).", internal_note="dup", confidence=0.8, escalate=True)], verbose=False)
assert o["category"] == "duplicate" and o["duplicate_of"] == "TKT-122022" and o["escalate"] is False
assert any(s["action"] == "validation_feedback" and "TKT-122022" in s["input"] for s in o["reasoning_trace"])

# 6) vague ticket -> needs_info forced to P3 + clarification
o = run("TKT-128253", [A("submit_triage", category="needs_info", priority="P1", route_to="L1_support", kb_references=["KB-2914"],
      customer_reply="Which module and error?", internal_note="vague", confidence=0.5)], verbose=False)
assert o["priority"] == "P3" and o["needs_clarification"] and not o["escalate"]

# 7) failure modes: no JSON -> valid fallback; endless tool use -> step cap
o = run("TKT-128253", ["broken!", "no json", "still no json"], verbose=False); assert o["confidence"] == 0.0
o = run("TKT-144939", [A("search_kb", query=f"ocr {i}") for i in range(12)], verbose=False); assert o["confidence"] == 0.0

# 8) tool facts on real data
T = CTX.tools
row = TICKETS.set_index("ticket_id").loc["TKT-169327"].to_dict() | {"ticket_id": "TKT-169327"}; T.bind(row)
r = T.call("resolve_account", {"domain": row["requester_domain"], "company_name": row["company_name_raw"]}); assert r["match"] == "ACC-4850", r
r = T.call("resolve_account", {"company_name": "Ironleaf Foods EU"}); assert "warning" in r or r["confidence"] == "low", r
row = TICKETS.set_index("ticket_id").loc["TKT-160094"].to_dict() | {"ticket_id": "TKT-160094"}; T.bind(row)
assert T.call("get_account_profile", {"account_id": "ACC-2507"})["sso_changed_within_30_days"] is True
r = T.call("search_kb", {"query": "invoice OCR failing", "error_code": "E-4102"}); assert r["results"][0]["id"] == "KB-3985", r["results"][0]
assert T.call("search_kb", {"query": "bulk approve 500 POs greyed out"})["results"][0]["id"] == "KB-4859"
print("\nALL SMOKE TESTS PASSED")
