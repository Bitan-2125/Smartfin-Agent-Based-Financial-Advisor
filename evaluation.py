"""Offline, repeatable engineering evaluation for the SmartFin portfolio project.

The score measures these explicit checks, not the quality of personal financial advice.
No API keys or network calls are required.
"""

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

os.environ.setdefault("SECRET_KEY", "evaluation-only-secret-key-32-characters-long")
os.environ.setdefault("USE_LLM", "false")

import app as webapp
from planning import build_plan, debt_payoff, future_value, required_monthly
from storage import load_plan


FORM = {
    "age": "30", "income": "75000", "expenses": "45000", "goal": "1000000",
    "risk": "Medium", "dependents": "1", "has_insurance": "none",
    "deduction_80c": "50000", "deduction_80d": "12000", "deduction_nps": "10000",
}


def require(condition, explanation):
    if not condition:
        raise AssertionError(explanation)


@contextmanager
def browser():
    with tempfile.TemporaryDirectory() as folder:
        old = webapp.app.config["DATABASE_PATH"]
        webapp.app.config["DATABASE_PATH"] = os.path.join(folder, "evaluation.sqlite3")
        try:
            client = webapp.app.test_client()
            require(client.get("/plan").status_code == 200, "planner did not open")
            with client.session_transaction() as session:
                csrf = session["csrf_token"]
            yield client, csrf
        finally:
            webapp.app.config["DATABASE_PATH"] = old


def generate(client, csrf, form=None):
    with patch.object(webapp, "llm", None):
        return client.post("/generate", data={**FORM, **(form or {}), "csrf_token": csrf})


def cash_flow():
    profile, goals, assumptions, errors = webapp.parse_plan_form({**FORM, "monthly_take_home": "65000", "monthly_emi": "5000", "liquid_savings": "30000", "investments": "50000", "debt_balance": "100000"})
    require(not errors, f"valid profile rejected: {errors}")
    plan = build_plan(profile, goals, assumptions)
    require(plan["monthly_surplus"] == 15000, "surplus differs from 65000 - 45000 - 5000")
    require(plan["net_worth"] == -20000, "net worth differs from 30000 + 50000 - 100000")


def emergency_reserve():
    profile, goals, assumptions, errors = webapp.parse_plan_form({**FORM, "monthly_emi": "5000", "liquid_savings": "30000", "emergency_months": "6"})
    require(not errors, f"valid profile rejected: {errors}")
    plan = build_plan(profile, goals, assumptions)
    require(plan["emergency_target"] == 300000, "six months of 50000 obligations should be 300000")
    require(plan["emergency_gap"] == 270000, "gap should subtract liquid savings")


def goal_math():
    require(future_value(1000, 500, 0, 12) == 7000, "zero-return future value")
    require(required_monthly(120000, 0, 0, 12) == 10000, "zero-return monthly contribution")
    require(required_monthly(1000, 1000, 0, 12) == 0, "already-funded goal")


def debt_math():
    require(debt_payoff(1000, 100, 0)["months"] == 10, "zero-interest payoff")
    require(debt_payoff(1200, 10, 12)["status"] == "payment_too_low", "interest-only payment should be flagged")


def tax_scope():
    require(webapp.tax_estimate(500000, 30) is None, "income above documented scope should not be estimated")
    require(webapp.tax_estimate(100000, 30)["deduction_80c"] == 0, "missing 80C should not be invented")


def invalid_inputs():
    require("income" in webapp.validate_profile({**FORM, "income": "NaN"}), "NaN accepted")
    require("risk" in webapp.validate_profile({**FORM, "risk": "Extreme"}), "unknown risk accepted")


def anonymous_entry():
    with browser() as (client, _):
        require(client.get("/login").status_code == 302, "old login link should redirect")
        require(client.get("/register").status_code == 404, "registration remains exposed")


def csrf_protection():
    with browser() as (client, _):
        require(client.post("/generate", data=FORM).status_code == 403, "plan creation lacks CSRF protection")


def session_isolation():
    with browser() as (client, csrf):
        require(generate(client, csrf).status_code == 302, "plan generation failed")
        other = webapp.app.test_client()
        require(other.get("/result").status_code == 302, "other browser saw the report")
        require(other.get("/export").status_code == 404, "other browser exported the report")
        require(client.get("/result").status_code == 200, "owner browser lost the report")


def deletion():
    with browser() as (client, csrf):
        require(generate(client, csrf).status_code == 302, "plan generation failed")
        with client.session_transaction() as session:
            token = session["token"]
        require(client.post("/delete", data={"csrf_token": csrf}).status_code == 302, "delete failed")
        require(load_plan(webapp.app.config["DATABASE_PATH"], token) is None, "plan still stored")
        require(client.get("/export").status_code == 404, "deleted plan still exports")


def ai_consent():
    with browser() as (client, csrf):
        with patch.object(webapp, "llm") as provider:
            require(client.post("/generate", data={**FORM, "csrf_token": csrf}).status_code == 302, "plan generation failed")
            provider.invoke.assert_not_called()
            require(client.post("/chat", json={"query": "hello"}, headers={"X-CSRF-Token": csrf}).status_code == 403, "chat accepted without consent")


def provider_fallback():
    with browser() as (client, csrf):
        with patch.object(webapp, "llm") as provider:
            provider.invoke.side_effect = RuntimeError("test provider outage")
            with patch.object(webapp.logger, "exception"):
                require(client.post("/generate", data={**FORM, "ai_consent": "1", "csrf_token": csrf}).status_code == 302, "provider outage broke plan creation")
            require(b"planning snapshot" in client.get("/result").data, "calculated fallback is missing")


CASES = [
    ("Planning accuracy", 8, "Cash flow and net worth", cash_flow),
    ("Planning accuracy", 8, "Emergency reserve", emergency_reserve),
    ("Planning accuracy", 8, "Goal arithmetic", goal_math),
    ("Planning accuracy", 8, "Debt payoff edge cases", debt_math),
    ("Planning accuracy", 8, "Tax estimator scope", tax_scope),
    ("Input quality", 10, "Nonfinite and invalid inputs", invalid_inputs),
    ("Privacy and control", 5, "Anonymous entry", anonymous_entry),
    ("Privacy and control", 5, "CSRF protection", csrf_protection),
    ("Privacy and control", 10, "Browser session isolation", session_isolation),
    ("Privacy and control", 10, "Saved data deletion", deletion),
    ("AI safeguards", 10, "Explicit AI consent", ai_consent),
    ("AI safeguards", 10, "Provider outage fallback", provider_fallback),
]


def run_evaluation():
    results = []
    for category, weight, name, check in CASES:
        try:
            check()
            passed, detail = True, ""
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append({"category": category, "name": name, "weight": weight, "passed": passed, "detail": detail})
    return {
        "score": sum(item["weight"] for item in results if item["passed"]),
        "maximum": sum(item["weight"] for item in results),
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "cases": results,
    }


def markdown(report):
    lines = [
        "# SmartFin evaluation", "",
        f"**Engineering score: {report['score']}/{report['maximum']}**; {report['passed']}/{report['total']} checks passing", "",
        "This is a repeatable local engineering score for the listed checks. It does not measure investment advice quality, future returns, AI factuality, or readiness for regulated financial advice.", "",
        "Run `python evaluation.py --markdown` to refresh this report. No paid services or API keys are used.", "",
        "| Category | Check | Weight | Result |", "| --- | --- | ---: | --- |",
    ]
    for item in report["cases"]:
        status = "Pass" if item["passed"] else f"Fail: {item['detail']}"
        lines.append(f"| {item['category']} | {item['name']} | {item['weight']} | {status} |")
    lines.extend(["", "The score is the sum of passing check weights; failed checks earn zero. The rubric is intentionally explicit so reviewers can inspect or extend it.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="print a GitHub-ready report")
    args = parser.parse_args()
    report = run_evaluation()
    print(markdown(report) if args.markdown else json.dumps(report, indent=2))
    raise SystemExit(0 if report["score"] == report["maximum"] else 1)
