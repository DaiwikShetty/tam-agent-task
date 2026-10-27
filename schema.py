"""Required output schema for each triaged ticket. Do not rename fields."""

CATEGORIES = ["incident", "bug", "how_to", "feature_request", "security", "duplicate", "needs_info"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
ROUTES = ["L1_support", "L2_engineering", "security", "product"]

EXAMPLE = {
    "ticket_id": "TKT-000000",
    "account_id": "ACC-0000",            # resolved account, or null if it cannot be resolved
    "category": "bug",                   # one of CATEGORIES
    "priority": "P3",                    # one of PRIORITIES - your decision, not the customer's
    "route_to": "L1_support",            # one of ROUTES
    "escalate": False,
    "notify_tam": False,
    "duplicate_of": None,                # ticket_id of the original ticket, or null
    "needs_clarification": False,
    "kb_references": ["KB-0000"],        # article ids that support the decision
    "account_signals": ["Renewal in 40 days"],  # facts from account/usage/ticket history that influenced the decision
    "customer_reply": "Draft reply to the customer.",
    "internal_note": "Short note for the support engineer.",
    "confidence": 0.8,                   # 0.0 - 1.0
    "reasoning_trace": [                 # one entry per step the agent took, in order
        {"step": 1, "action": "lookup_account", "input": "requester_domain=...", "observation": "Found ACC-0000, Premium"},
    ],
}

_TYPES = {"ticket_id": str, "category": str, "priority": str, "route_to": str, "escalate": bool, "notify_tam": bool,
          "needs_clarification": bool, "kb_references": list, "account_signals": list, "customer_reply": str,
          "internal_note": str, "confidence": (int, float), "reasoning_trace": list}


def validate(out: dict) -> list[str]:
    """Return a list of problems; empty list means valid."""
    errs = []
    for k, t in _TYPES.items():
        if k not in out:
            errs.append(f"missing field: {k}")
        elif not isinstance(out[k], t) or (t is not bool and isinstance(out[k], bool) and k == "confidence"):
            errs.append(f"wrong type for {k}")
    for k in ("account_id", "duplicate_of"):
        if k not in out:
            errs.append(f"missing field: {k}")
        elif out[k] is not None and not isinstance(out[k], str):
            errs.append(f"{k} must be string or null")
    if out.get("category") not in CATEGORIES:
        errs.append(f"category must be one of {CATEGORIES}")
    if out.get("priority") not in PRIORITIES:
        errs.append(f"priority must be one of {PRIORITIES}")
    if out.get("route_to") not in ROUTES:
        errs.append(f"route_to must be one of {ROUTES}")
    c = out.get("confidence")
    if isinstance(c, (int, float)) and not isinstance(c, bool) and not 0 <= c <= 1:
        errs.append("confidence must be between 0 and 1")
    trace = out.get("reasoning_trace")
    if isinstance(trace, list):
        if not trace:
            errs.append("reasoning_trace is empty")
        for s in trace:
            if not isinstance(s, dict) or not {"step", "action", "observation"} <= set(s):
                errs.append("each reasoning_trace entry needs step, action, observation")
                break
    return errs
