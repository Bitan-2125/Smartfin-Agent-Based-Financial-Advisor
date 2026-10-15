# SmartFin evaluation

**Engineering score: 100/100**; 12/12 checks passing

This is a repeatable local engineering score for the listed checks. It does not measure investment advice quality, future returns, AI factuality, or readiness for regulated financial advice.

Run `python evaluation.py --markdown` to refresh this report. No paid services or API keys are used.

| Category | Check | Weight | Result |
| --- | --- | ---: | --- |
| Planning accuracy | Cash flow and net worth | 8 | Pass |
| Planning accuracy | Emergency reserve | 8 | Pass |
| Planning accuracy | Goal arithmetic | 8 | Pass |
| Planning accuracy | Debt payoff edge cases | 8 | Pass |
| Planning accuracy | Tax estimator scope | 8 | Pass |
| Input quality | Nonfinite and invalid inputs | 10 | Pass |
| Privacy and control | Anonymous entry | 5 | Pass |
| Privacy and control | CSRF protection | 5 | Pass |
| Privacy and control | Browser session isolation | 10 | Pass |
| Privacy and control | Saved data deletion | 10 | Pass |
| AI safeguards | Explicit AI consent | 10 | Pass |
| AI safeguards | Provider outage fallback | 10 | Pass |

The score is the sum of passing check weights; failed checks earn zero. The rubric is intentionally explicit so reviewers can inspect or extend it.
