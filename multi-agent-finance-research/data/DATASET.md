# Dataset card — `companies.csv`

## What this is
An **illustrative, compiled snapshot** of 25 large publicly-traded companies, used
to demonstrate the **Data-Analyst agent's** tool use (filtering, aggregation,
ranking, ratio computation). It is *not* a live market feed.

## Columns
| Column             | Type    | Meaning                                              |
|--------------------|---------|------------------------------------------------------|
| `company`          | string  | Common company name                                  |
| `ticker`           | string  | Primary stock ticker                                 |
| `sector`           | string  | GICS-style sector label                              |
| `country`          | string  | Country of headquarters                              |
| `founded`          | int     | Year founded                                         |
| `employees`        | int     | Approximate headcount                                |
| `revenue_usd_b`    | float   | Approximate annual revenue, USD billions             |
| `net_income_usd_b` | float   | Approximate annual net income, USD billions          |
| `market_cap_usd_b` | float   | Approximate market capitalization, USD billions      |

## Provenance & honesty disclaimer
Figures are **approximate**, hand-compiled to the ballpark of recent (~FY2023)
public reporting for educational use. They are rounded, may be stale, and **must
not be used for investment decisions**. The project's value is the multi-agent
*architecture* and *evaluation methodology*, not the authority of these numbers.

For *current* figures the system intentionally has a separate **Web-Research
agent** that performs live lookups — this division of labour (static curated
facts vs. live facts) is itself a deliberate design decision (see the README).

## Why a curated CSV instead of a live financial API
A bundled CSV keeps the repository **fully reproducible** with no paid API keys,
and lets us build an **objective, deterministic ground truth** for the
data-analyst evaluation: every "what is the highest market-cap company?"-style
question has one verifiable answer computed directly from this file.
