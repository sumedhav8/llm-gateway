"""Report from the requests table, broken down by route.

Shows, for each route (direct cheap/strong, cascade-accepted, cascade-escalated,
router-cheap, ...): how many requests, what they cost, and how long they took.

Usage (from the project folder; the database must be running):
    python -m scripts.route_report              # last 24 hours
    python -m scripts.route_report --hours 2
"""

import argparse
import os

import psycopg

from app import config  # noqa: F401  (importing config loads the .env file)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gateway:gateway_dev_password@localhost:5432/gateway")

QUERY = """
SELECT
    route,
    COUNT(*)                                                      AS requests,
    COALESCE(SUM(cost_usd), 0)                                    AS total_cost,
    COALESCE(AVG(cost_usd), 0)                                    AS avg_cost,
    COALESCE(AVG(latency_ms), 0)                                  AS avg_ms,
    COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95_ms
FROM requests
WHERE status_code = 200
  AND created_at >= NOW() - make_interval(hours => %s)
GROUP BY route
ORDER BY requests DESC
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=int, default=24, help="How far back to look")
    args = parser.parse_args()

    with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
        rows = conn.execute(QUERY, (args.hours,)).fetchall()

    if not rows:
        print(f"No successful requests in the last {args.hours} hours.")
        return

    print(f"Successful requests in the last {args.hours} hours, by route:\n")
    print(f"{'route':<38} {'requests':>8} {'total $':>10} {'avg $':>10} {'avg ms':>8} {'p95 ms':>8}")
    for route, requests, total, avg, avg_ms, p95_ms in rows:
        print(f"{route:<38} {requests:>8} {float(total):>10.4f} {float(avg):>10.6f} "
              f"{float(avg_ms):>8,.0f} {float(p95_ms):>8,.0f}")
    total_requests = sum(row[1] for row in rows)
    total_cost = sum(float(row[2]) for row in rows)
    print(f"\n{'all routes':<38} {total_requests:>8} {total_cost:>10.4f}")


if __name__ == "__main__":
    main()