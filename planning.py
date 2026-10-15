"""Deterministic, editable financial planning scenarios.

All money inputs are rupees. Rates are annual percentages. This module does not
select financial products or predict investment returns.
"""

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP


ZERO = Decimal("0")
ONE = Decimal("1")


def amount(value):
    return Decimal(str(value))


def rupees(value):
    return int(value.quantize(ONE, rounding=ROUND_HALF_UP))


def monthly_rate(annual_percent):
    return (ONE + amount(annual_percent) / 100) ** (ONE / 12) - ONE


def future_value(initial, monthly_contribution, annual_percent, months):
    """Value when the contribution is made at the end of each month."""
    rate = monthly_rate(annual_percent)
    growth = (ONE + rate) ** months
    contributions = amount(monthly_contribution) * months if rate == 0 else amount(monthly_contribution) * (growth - ONE) / rate
    return amount(initial) * growth + contributions


def required_monthly(target, initial, annual_percent, months):
    """Whole-rupee contribution needed to reach a future target."""
    if months <= 0:
        return 0 if amount(initial) >= amount(target) else None
    rate = monthly_rate(annual_percent)
    growth = (ONE + rate) ** months
    unfunded = max(ZERO, amount(target) - amount(initial) * growth)
    factor = amount(months) if rate == 0 else (growth - ONE) / rate
    return int((unfunded / factor).quantize(ONE, rounding=ROUND_CEILING))


def debt_payoff(balance, monthly_payment, annual_percent):
    """Illustrative payoff for one fixed-rate amortizing debt, capped at 50 years."""
    balance = amount(balance)
    payment = amount(monthly_payment)
    rate = amount(annual_percent) / 1200
    if balance <= 0:
        return {"months": 0, "total_interest": 0, "status": "no_debt"}
    if payment <= balance * rate:
        return {"months": None, "total_interest": None, "status": "payment_too_low"}
    interest_paid = ZERO
    for month in range(1, 601):
        interest = balance * rate
        interest_paid += interest
        balance = balance + interest - payment
        if balance <= 0:
            return {"months": month, "total_interest": rupees(interest_paid), "status": "paying_off"}
    return {"months": None, "total_interest": None, "status": "over_50_years"}


def build_plan(profile, goals, assumptions):
    """Build a plan from validated values without paid services or market data."""
    income = amount(profile.get("cash_income", profile["income"]))
    expenses = amount(profile["expenses"])
    emi = amount(profile["monthly_emi"])
    liquid = amount(profile["liquid_savings"])
    invested = amount(profile["investments"])
    debt = amount(profile["debt_balance"])
    surplus = income - expenses - emi
    emergency_months = int(profile["emergency_months"])
    emergency_target = (expenses + emi) * emergency_months
    emergency_gap = max(ZERO, emergency_target - liquid)
    life_target = amount(profile["life_target"])
    health_target = amount(profile["health_target"])
    life_gap = max(ZERO, life_target - amount(profile["life_cover"]))
    health_gap = max(ZERO, health_target - amount(profile["health_cover"]))

    scenarios = ("low", "base", "high")
    goal_results = []
    for goal in goals:
        months = int(goal["years"]) * 12
        future_target = amount(goal["amount"]) * (ONE + amount(assumptions["inflation"]) / 100) ** int(goal["years"])
        scenario_results = {}
        for name in scenarios:
            rate = assumptions[f"return_{name}"]
            required = required_monthly(future_target, goal["saved"], rate, months)
            projected = future_value(goal["saved"], goal["monthly"], rate, months)
            scenario_results[name] = {
                "required_monthly": required,
                "projected_value": rupees(projected),
                "shortfall": rupees(max(ZERO, future_target - projected)),
                "on_track": projected >= future_target,
            }
        goal_results.append({
            "name": goal["name"],
            "today_cost": rupees(amount(goal["amount"])),
            "years": int(goal["years"]),
            "saved": rupees(amount(goal["saved"])),
            "planned_monthly": rupees(amount(goal["monthly"])),
            "future_target": rupees(future_target),
            "scenarios": scenario_results,
        })

    required_base = sum(g["scenarios"]["base"]["required_monthly"] for g in goal_results)
    planned_total = sum(g["planned_monthly"] for g in goal_results)
    available_for_emergency = max(ZERO, surplus - amount(planned_total))
    emergency_fill_months = (
        int((emergency_gap / available_for_emergency).quantize(ONE, rounding=ROUND_CEILING))
        if emergency_gap > 0 and available_for_emergency > 0 else None
    )
    priorities = []
    if surplus < 0:
        priorities.append("Monthly obligations exceed income. Review spending and debt payments before adding investments.")
    if emergency_gap > 0:
        priorities.append("Build an accessible emergency reserve before relying on long-term investments for short-term needs.")
    if debt > 0 and amount(profile["debt_rate"]) >= 12:
        priorities.append("Review the high-interest debt and compare repayment with other uses of cash.")
    if int(profile["dependents"]) > 0 and life_gap > 0:
        priorities.append("Review the life-cover gap with a licensed insurance professional.")
    if health_gap > 0:
        priorities.append("Review health-cover adequacy and policy terms.")
    if required_base > max(ZERO, surplus):
        priorities.append("The goals require more monthly funding than the current surplus; adjust timelines, amounts, or cash flow.")
    if not priorities:
        priorities.append("Review assumptions and rebalance the plan when income, expenses, or goals change.")

    return {
        "monthly_income": rupees(income),
        "monthly_expenses": rupees(expenses),
        "monthly_emi": rupees(emi),
        "monthly_surplus": rupees(surplus),
        "savings_rate_percent": round(float(surplus / income * 100), 1) if income else 0,
        "emi_to_income_percent": round(float(emi / income * 100), 1) if income else 0,
        "net_worth": rupees(liquid + invested - debt),
        "emergency_target": rupees(emergency_target),
        "emergency_gap": rupees(emergency_gap),
        "emergency_coverage_months": round(float(liquid / (expenses + emi)), 1) if expenses + emi else None,
        "emergency_fill_months": emergency_fill_months,
        "life_target": rupees(life_target),
        "life_gap": rupees(life_gap),
        "health_target": rupees(health_target),
        "health_gap": rupees(health_gap),
        "debt_payoff": debt_payoff(debt, emi, profile["debt_rate"]),
        "required_goal_monthly": required_base,
        "planned_goal_monthly": planned_total,
        "goals_affordable": required_base <= max(ZERO, surplus),
        "goals": goal_results,
        "assumptions": assumptions,
        "priorities": priorities,
    }
