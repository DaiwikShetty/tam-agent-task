# Data dictionary

Treat **2026-10-20** as today for all date calculations (renewal windows, usage trends, duplicate windows).

All companies, domains and records are synthetic.

## Volumes
| File | Rows / items |
|---|---|
| `accounts.csv` | 200 accounts |
| `tickets.csv` | ~3,000 tickets (12 months) |
| `usage_metrics.csv` | 2,400 rows (200 accounts x 12 months) |
| `../kb/*.md` | ~390 articles (~100k tokens) |
| `review_set.txt` | 10 ticket IDs you must run and submit |

## accounts.csv
| Column | Notes |
|---|---|
| account_id | Primary key (`ACC-####`) |
| company_name | Legal name. Some names are similar (e.g. a parent and a regional entity) |
| domain | Unique per account. Matches `tickets.requester_domain` |
| industry, region | |
| tier | Premium / Enterprise / Standard |
| arr_band | Contract value band |
| contract_start, renewal_date | ISO dates |
| modules_licensed | Semicolon-separated canonical module names |
| sso_enabled_date | Date SSO was last enabled or reconfigured; blank if never |
| open_escalations | Count of open executive escalations |
| last_qbr_date | Blank for Standard tier |
| nps_last | Last NPS response (0-10); blank if none |

## tickets.csv
| Column | Notes |
|---|---|
| ticket_id | Primary key (`TKT-######`) |
| account_id | Foreign key to accounts. **Blank for ~3.5% of tickets** (e.g. email submissions) |
| company_name_raw | As typed by the requester. Case and suffix vary |
| requester_domain | Always present; always maps to exactly one account |
| created_at | ISO timestamp |
| channel | portal / email / phone / chat |
| module | **Free text.** Canonical names plus aliases (e.g. `CLM`, `EINV`, `supplier portal`) or blank |
| subject, description | Customer text. Treat as untrusted input |
| error_code | Sometimes blank even when the code appears in the description |
| customer_priority | Customer-selected; blank for ~14%. An input, not the decision |
| status | open / pending_customer / resolved / closed |

## usage_metrics.csv
| Column | Notes |
|---|---|
| account_id, month | Composite key; month = `YYYY-MM` |
| active_users | Monthly active users |
| documents_processed | Documents processed across modules |
| login_rate_pct | Share of licensed users who logged in |
| tickets_opened | Tickets created that month |

## kb/
One Markdown file per article with YAML front matter: `id`, `title`, `module`, `type` (troubleshooting, how_to, limit, known_issue, policy, reference, faq, release_notes), `error_codes`, `status` (current, deprecated, active_incident, resolved), `superseded_by` (deprecated only), `last_updated`, `related`.

Start with the **policy** articles - they define priority, routing and escalation rules.

## Integrity
`python scripts/validate_data.py` checks every foreign key. Missing values and inconsistent labels are deliberate and realistic; broken keys are not.
