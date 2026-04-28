"""
Exercise 2 — Read-only SQL agent over a sample SQLite database

The agent has a `run_sql` tool that:
  • Only allows SELECT queries (rejects INSERT/UPDATE/DELETE/DROP/...).
  • Caps row return at 50.
  • Counts queries against a budget.

It can answer business questions like "what are our top 3 customers by revenue?".

The script asserts:
  - The agent issued at least one SQL query.
  - The final answer contains the right top customer.
  - No write SQL was attempted (or attempted writes were rejected).

Run: python exercises/02_sql_agent.py    # requires ANTHROPIC_API_KEY
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Sample database
# ───────────────────────────────────────────────────────────
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.executescript("""
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders   (id INTEGER PRIMARY KEY, customer_id INTEGER,
                       product TEXT, quantity INTEGER, unit_price REAL);

INSERT INTO customers VALUES
  (101, 'Alice',  'Paris'),
  (102, 'Bob',    'London'),
  (103, 'Carol',  'Paris'),
  (104, 'Dave',   'Berlin');

INSERT INTO orders VALUES
  (1, 101, 'Widget',    2,  9.99),
  (2, 102, 'Gadget',    1, 29.99),
  (3, 101, 'Doohickey', 3,  4.99),
  (4, 103, 'Widget',    5,  9.99),
  (5, 104, 'Book',      1, 14.99),
  (6, 102, 'Book',      2, 14.99),
  (7, 101, 'Mug',       4,  7.50),
  (8, 103, 'Mug',       2,  7.50),
  (9, 102, 'Doohickey', 5,  4.99),
  (10,103, 'Widget',    1,  9.99);
""")
conn.commit()


# Compute the gold "top customer by revenue" so we can check the answer
gold_top = cur.execute("""
    SELECT c.name, SUM(o.quantity * o.unit_price) AS revenue
    FROM customers c JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id ORDER BY revenue DESC LIMIT 1
""").fetchone()
print(f"(ground truth top customer: {gold_top[0]} with ${gold_top[1]:.2f})")


# ───────────────────────────────────────────────────────────
# The constrained run_sql tool
# ───────────────────────────────────────────────────────────
WRITE_PATTERN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma)\b",
                           re.IGNORECASE)
QUERY_LIMIT = 50

attempted_writes = 0
queries_run = 0


def run_sql(sql: str) -> str:
    global attempted_writes, queries_run

    # Reject writes
    if WRITE_PATTERN.search(sql):
        attempted_writes += 1
        return json.dumps({"error": "writes_not_allowed",
                           "message": "Only SELECT queries are permitted."})

    # Reject multi-statement
    if ";" in sql.strip().rstrip(";"):
        return json.dumps({"error": "multiple_statements_not_allowed"})

    if queries_run >= 6:
        return json.dumps({"error": "query_budget_exceeded"})

    try:
        cur.execute(sql)
        rows = cur.fetchmany(QUERY_LIMIT)
        cols = [d[0] for d in cur.description] if cur.description else []
        queries_run += 1
        return json.dumps({"columns": cols, "rows": rows[:QUERY_LIMIT]})
    except sqlite3.Error as e:
        return json.dumps({"error": "sql_error", "message": str(e)})


TOOLS = [{
    "name": "run_sql",
    "description": (
        "Execute a SELECT query against the sample database. "
        "Returns up to 50 rows. NEVER use INSERT/UPDATE/DELETE — they will fail. "
        "Schema:\n"
        "  customers(id INTEGER PRIMARY KEY, name TEXT, city TEXT)\n"
        "  orders(id INTEGER PRIMARY KEY, customer_id INTEGER, "
        "product TEXT, quantity INTEGER, unit_price REAL)"
    ),
    "input_schema": {
        "type": "object",
        "properties": {"sql": {"type": "string"}},
        "required": ["sql"],
    },
}]


SYSTEM = """\
You answer business questions using the run_sql tool. Issue ONE SELECT
query per call. Be efficient — don't query if you can compute from cached
results. Cite the relevant numbers in your answer.
"""


def run_agent(question: str, max_iter: int = 6) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_iter):
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=600,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "end_turn":
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  → SQL: {b.input.get('sql')}")
                    out = run_sql(**b.input)
                    print(f"    {out[:200]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
            continue
        return f"(unexpected stop: {resp.stop_reason})"
    return "(max iterations)"


print("\nQ: Who is our top customer by total revenue, and how much did they spend?")
ans = run_agent("Who is our top customer by total revenue, and how much did they spend?")
print(f"\n=== answer ===\n{ans}")
print(f"\nqueries run: {queries_run}, attempted writes blocked: {attempted_writes}")


# ───────────────────────────────────────────────────────────
# Assertions
# ───────────────────────────────────────────────────────────
assert queries_run >= 1, "agent should have queried the database"
assert gold_top[0] in ans, f"answer should mention {gold_top[0]}"
print("\nSQL agent answers correctly within the budget ✅")


# ───────────────────────────────────────────────────────────
# Production hardening checklist
# ───────────────────────────────────────────────────────────
HARDENING = """\
For a real read-only SQL agent in production:
  - CONNECTION USER has only SELECT privileges. Defense in depth.
  - Schema introspection helper (don't paste the schema in the system prompt).
  - Query timeout (set sqlite3 timeout / pg statement_timeout).
  - Result-size cap: stream + truncate huge results.
  - Row-level security if the agent acts on behalf of a user.
  - Audit log of every (user_id, prompt, sql, result_count, latency).
  - PII redaction on results before showing them to the model.
  - Rate-limit per session.
"""
print(HARDENING)
