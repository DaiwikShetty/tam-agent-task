# SOLUTION

## 1. Design
`triage(ticket, ctx)` is a tool-using loop. Each turn the LLM emits **one JSON action** (`{"thought","tool","args"}`), reads the observation and picks the next action. It stops by calling `submit_triage`; a 10-turn cap forces a submit. There is no fixed pipeline: a how-to may need 2 lookups, a suspected incident 6-8.

**Tools** (indexes built once in `Context`): `resolve_account` (requester domain is authoritative; fuzzy name is a flagged fallback), `get_account_profile` (tier, renewal days, escalations, recent SSO change, module licensed, inputs of the KB-2914 risk rule), `get_usage_trend`, `search_tickets` (account history, KB-5930 duplicate flag, same error code across accounts as an incident signal), `list_policies`, `search_kb` (stemmed BM25 + error-code boost; deprecated articles down-ranked), `read_kb_article` (full text, related articles, deprecated/incident warnings), `submit_triage`. Module aliases are normalised and error codes in the description are pre-extracted. The prompt lists policy titles so safety-critical policy is always discoverable.

**Guardrails.** Ticket text is fenced as untrusted; instruction-like text is flagged and ignored. A finalizer checks that the account matches the domain, KB ids and `duplicate_of` exist, and strips other customers' data from replies. Mechanical policy consequences are enforced (security -> P1/security/escalate, P1 -> escalate, risk override -> notify TAM, insufficient info -> P3, duplicates not re-escalated); judgement calls (a missed duplicate, applying the override) go back to the model once. Unusable output falls back to a schema-valid "manual triage" record. The loop writes `reasoning_trace` itself.Replaced the urllib transport with PowerShell Invoke-RestMethod because urllib stalled in my environment.

## 2. Evaluation
- ~300 tickets labelled by senior support, stratified by category/tier/ambiguity; per-field accuracy, weighting **priority, security and escalation recall** highest.
- Priority confusion matrix and under-triage rate (agent lower than gold).
- Retrieval recall@k of the gold article; rate of citing deprecated articles.
- Duplicate precision/recall; account resolution on blank-id and similar-name cases.
- Adversarial suite (embedded instructions, fake urgency, polite security reports): decisions unchanged, no cross-customer leakage.
- Trace audit for unsupported claims; track steps, tokens, latency and fallback rate.
- Shadow mode vs human triage for 2-4 weeks, then human review of low-confidence, P1 and security outputs.

## 3. Scale (10x)
Per-ticket LLM cost is bounded by the step cap and small observations; total cost grows linearly with volume. BM25 stays fast; the ticket scan is O(N) and should move to an inverted index or database. The real risk is **retrieval quality**: more near-duplicate and outdated articles. I would add hybrid BM25 + embeddings with a reranker, collapse superseded chains at index time, cache policy text in the prompt prefix, and use a cheaper model for routine tickets with escalation on low confidence.

## 4. Limitations
- Retrieval hand-checked on the review tickets only: expected article in the top 5 for 17 of 20 lookups; misses were vague queries.
- The policy guard hard-codes KB-2914/5930/6401 rules and must be kept in sync with policy.
- Plumbing is tested with a scripted LLM (`tests/smoke_test.py`); decision quality is **not yet measured**. [TODO: add review-set results after running a real model.]
- Lexical retrieval misses paraphrases; the injection detector is a regex backstop; JSON actions (not native tool calling) trade efficiency for provider portability.

## 5. How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # + anthropic or openai if used
cp .env.example .env              # set LLM_PROVIDER / LLM_MODEL / key
python scripts/validate_data.py
python run.py --set review
python scripts/check_outputs.py   # must show 10/10 valid
python tests/smoke_test.py        # plumbing test, no LLM needed
```
