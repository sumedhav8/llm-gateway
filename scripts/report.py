"""Print a summary of every request the gateway has handled."""

import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

DEFAULT_DATABASE_URL = "postgresql://gateway:gateway_dev_password@localhost:5432/gateway"

SUMMARY_SQL = """
    SELECT
        COUNT(*)                                        AS total,
        COUNT(*) FILTER (WHERE status_code = 200)       AS successful,
        ROUND(AVG(latency_ms) FILTER (WHERE status_code = 200))
                                                        AS avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE status_code = 200)            AS p95_latency_ms,
        MAX(latency_ms) FILTER (WHERE status_code = 200)
                                                        AS max_latency_ms,
        COALESCE(SUM(input_tokens), 0)                  AS input_tokens,
        COALESCE(SUM(output_tokens), 0)                 AS output_tokens,
        COALESCE(SUM(cost_usd), 0)                      AS total_cost_usd
    FROM requests
"""

BY_MODEL_SQL = """
    SELECT
        model,
        COUNT(*)                                        AS total,
        ROUND(AVG(latency_ms) FILTER (WHERE status_code = 200))
                                                        AS avg_latency_ms,
        COALESCE(SUM(cost_usd), 0)                      AS total_cost_usd
    FROM requests
    GROUP BY model
    ORDER BY total DESC
"""


def ms(value):
    return "-" if value is None else f"{int(value):,} ms"


def main():
    load_dotenv()
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    with psycopg.connect(url, row_factory=dict_row) as conn:
        s = conn.execute(SUMMARY_SQL).fetchone()
        models = conn.execute(BY_MODEL_SQL).fetchall()

    success_rate = 100 * s["successful"] / s["total"] if s["total"] else 0

    print("Gateway report")
    print("==============")
    print(f"Requests:       {s['total']}  ({s['successful']} successful)")
    print(f"Success rate:   {success_rate:.1f}%")
    print(f"Avg latency:    {ms(s['avg_latency_ms'])}")
    print(f"p95 latency:    {ms(s['p95_latency_ms'])}")
    print(f"Slowest:        {ms(s['max_latency_ms'])}")
    print(f"Tokens:         {s['input_tokens']:,} in / {s['output_tokens']:,} out")
    print(f"Total cost:     ${s['total_cost_usd']:.6f}")
    print()
    print("By model")
    print("--------")
    for m in models:
        print(f"{m['model']:<15} {m['total']:>4} requests   "
              f"avg {ms(m['avg_latency_ms']):>10}   ${m['total_cost_usd']:.6f}")


if __name__ == "__main__":
    main()