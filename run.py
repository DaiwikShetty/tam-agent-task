"""Run the triage agent.

  python run.py --set review            # the 10 review tickets (required)
  python run.py --set sample --n 50     # random sample (optional)
  python run.py --ids TKT-123456 TKT-234567

Writes one JSON per ticket to outputs/ plus outputs/run_log.jsonl.
"""
import argparse, json, random, time, traceback
from pathlib import Path

from tam_agent.data import load_accounts, load_tickets, load_usage, load_review_set
from tam_agent.kb import load_kb
from tam_agent.agent import Context, triage
from tam_agent.schema import validate

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["review", "sample"], default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    tickets = load_tickets()
    ctx = Context(load_accounts(), tickets, load_usage(), load_kb())
    if args.ids:
        ids = args.ids
    elif args.set == "sample":
        ids = random.Random(args.seed).sample(list(tickets["ticket_id"]), args.n)
    else:
        ids = load_review_set()

    OUT.mkdir(exist_ok=True)
    by_id = tickets.set_index("ticket_id")
    ok = 0
    with open(OUT / "run_log.jsonl", "a") as log:
        for tid in ids:
            t0, calls0 = time.time(), ctx.llm.calls
            rec = {"ticket_id": tid}
            try:
                ticket = by_id.loc[tid].to_dict() | {"ticket_id": tid}
                result = triage(ticket, ctx)
                errs = validate(result)
                (OUT / f"{tid}.json").write_text(json.dumps(result, indent=2))
                rec |= {"status": "ok" if not errs else "schema_error", "errors": errs}
                ok += not errs
            except Exception as e:
                rec |= {"status": "exception", "errors": [repr(e)], "traceback": traceback.format_exc(limit=3)}
            rec |= {"seconds": round(time.time() - t0, 2), "llm_calls": ctx.llm.calls - calls0}
            log.write(json.dumps(rec) + "\n")
            print(f"{tid}: {rec['status']} ({rec['seconds']}s, {rec['llm_calls']} LLM calls)")
    print(f"\n{ok}/{len(ids)} tickets produced valid output")


if __name__ == "__main__":
    main()
