import os
import uuid
import math
import logging
import secrets
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import — avoids Tkinter thread issues
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
from werkzeug.middleware.proxy_fix import ProxyFix
import io
import base64

from dotenv import load_dotenv
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from storage import (
    save_plan, load_plan, load_history, save_exchange, delete_plan,
    allow_anonymous_usage,
)
from prompts import SYSTEM_PROMPT
from planning import build_plan

# -----------------------
# ENV
# -----------------------
load_dotenv()

app = Flask(__name__)
trusted_proxies = int(os.getenv("TRUST_PROXY_COUNT", "0"))
if trusted_proxies:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=trusted_proxies, x_proto=trusted_proxies)
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key or len(app.secret_key) < 32 or app.secret_key == "replace_with_a_long_random_secret":
    raise RuntimeError("Set a random SECRET_KEY of at least 32 characters before starting SmartFin.")
app.config["DATABASE_PATH"] = os.getenv("DATABASE_URL") or os.getenv(
    "SMARTFIN_DB_PATH", os.path.join(app.instance_path, "smartfin.sqlite3")
)
app.config.update(
    MAX_CONTENT_LENGTH=64 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "true" if os.getenv("RENDER") or os.getenv("VERCEL") else "false").lower() == "true",
)
app.permanent_session_lifetime = timedelta(days=7)
logger = logging.getLogger(__name__)


def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def check_csrf():
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        submitted = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not expected or not secrets.compare_digest(submitted, expected):
            abort(403)


@app.after_request
def secure_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


groq_key = os.getenv("GROQ_API_KEY")
llm = (ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3,
) if groq_key and os.getenv("USE_LLM", "true").lower() == "true" else None)

tavily_key = os.getenv("TAVILY_API_KEY")
search_tool = (TavilySearch(
    api_key=tavily_key,
    max_results=5,
    topic="finance",
) if tavily_key else None)

# -----------------------
# CALCULATIONS AND AGENT FLOW
# -----------------------
def monthly_savings(income, expenses, emi=0):
    return income - expenses - emi

def sip_projection(monthly, months, annual_return=0.10, initial=0):
    r = (1 + annual_return) ** (1 / 12) - 1
    bal = initial
    values = []
    for _ in range(months):
        bal = bal * (1 + r) + monthly
        values.append(bal)
    return values

class FinanceState(TypedDict):
    age: int
    income: float
    cash_income: float
    expenses: float
    goal_amount: float
    risk: str
    monthly_emi: float
    primary_goal: dict
    planning_summary: str
    ai_allowed: bool
    base_return: float
    monthly_saving: float
    months_to_goal: Optional[int]
    growth: list
    chat_history: list
    user_query: str
    response: str
    search_result: str

def analyze(state):
    state["monthly_saving"] = monthly_savings(
        state["cash_income"], state["expenses"], state["monthly_emi"]
    )
    return state

def calculate(state):
    goal = state["primary_goal"]
    months = goal["years"] * 12
    state["growth"] = sip_projection(
        goal["monthly"], months, state["base_return"] / 100, goal["saved"]
    )
    state["months_to_goal"] = months if state["growth"][-1] >= goal["future_target"] else None
    return state

def search(state):
    q = state.get("user_query", "").lower()
    if llm is None or not state.get("ai_allowed"):
        state["search_result"] = ""
        return state
    if any(k in q for k in ["stock", "market", "invest", "nifty", "sensex", "mutual fund", "sip", "nse", "bse"]):
        if search_tool is None:
            state["search_result"] = "Current market search is unavailable. Do not imply live data was checked."
            return state
        try:
            result = search_tool.run(state["user_query"])
            if not isinstance(result, dict) or "results" not in result:
                raise ValueError("Search provider returned no results field")
            summaries = [
                f"{r.get('title', 'Source')} ({r.get('url', 'no URL')}): {r.get('content', '')}"
                for r in result["results"] if isinstance(r, dict)
            ]
            state["search_result"] = "\n".join(summaries) or "No current sources found."
        except Exception:
            logger.exception("Market search failed")
            state["search_result"] = "Current market search is unavailable. Do not imply live data was checked."
    else:
        state["search_result"] = ""
    return state

def advisor(state):
    initial_plan = state.get("user_query") == "Generate a detailed financial plan"
    writing_brief = (
        "Write a detailed Markdown financial planning explanation with clear sections: financial snapshot, "
        "cash-flow and emergency reserve, debt and protection, goal scenarios and tradeoffs, a sequenced "
        "action checklist, assumptions and uncertainties, and useful follow-up questions. Interpret the "
        "calculated figures rather than repeating a raw list. Discuss low/base/high return assumptions as "
        "hypothetical scenarios; only quote scenario values that are supplied. Make the plan specific to the "
        "profile, while keeping it educational rather than recommending products or transactions."
        if initial_plan else
        "Answer the follow-up question directly in Markdown. Use the saved conversation and the relevant "
        "calculated figures, explain tradeoffs and practical next steps, and acknowledge missing inputs. "
        "Do not repeat the full initial plan unless the user asks for it."
    )
    prompt = f"""
User Profile (Indian investor):
Age: {state['age']}
Monthly cash-flow surplus: ₹{state['monthly_saving']:,.0f}
Risk Tolerance: {state['risk']}
Calculated planning scenarios:
{state['planning_summary']}

Current search context: {state.get('search_result','')}
User Query: {state.get('user_query','')}

{writing_brief}
Do not invent new figures or specific product recommendations. Use ₹ for all amounts.
"""
    def calculated_response():
        return (
            "### Your planning snapshot\n\n" + state["planning_summary"]
            + "\n\nThese are editable scenarios, not predictions. AI commentary is unavailable; the calculated dashboard remains available."
        )
    if llm is None or not state.get("ai_allowed"):
        state["response"] = calculated_response()
        return state
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in state.get("chat_history", []):
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=prompt))
    try:
        res = llm.invoke(messages)
        state["response"] = res.content
    except Exception:
        logger.exception("AI commentary failed; using calculated response")
        state["response"] = calculated_response()
    return state

graph = StateGraph(FinanceState)
graph.add_node("analyze", analyze)
graph.add_node("calculate", calculate)
graph.add_node("search", search)
graph.add_node("advisor", advisor)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "calculate")
graph.add_edge("calculate", "search")
graph.add_edge("search", "advisor")
graph.add_edge("advisor", END)

smartfin = graph.compile()

# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return render_template("landing.html", has_plan=bool(session.get("token")))


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/login")
def old_login_link():
    return redirect(url_for("result_page") if session.get("token") else url_for("plan"))

@app.route("/plan")
def plan():
    return render_template("index.html", errors={}, values={}, extra_rows=[])


@app.route("/edit")
def edit_plan():
    data = load_plan(app.config["DATABASE_PATH"], session.get("token"))
    if data is None:
        return redirect(url_for("plan"))
    profile = data["profile"]
    primary = data["goals"][0]
    values = {
        **profile,
        "goal": primary["amount"],
        "goal_name": primary["name"],
        "goal_years": primary["years"],
        "goal_saved": primary["saved"],
        "goal_monthly": primary["monthly"],
        "monthly_take_home": profile.get("cash_income", profile["income"]),
        "ai_consent": "1" if data.get("ai_consent") else "",
        **data["assumptions"],
    }
    return render_template("index.html", errors={}, values=values, extra_rows=data["goals"][1:])

# -----------------------
# HELPERS
# -----------------------
def extra_goal_rows(data):
    keys = ("name", "amount", "years", "saved", "monthly")
    columns = {key: data.getlist(f"extra_goal_{key}") for key in keys} if hasattr(data, "getlist") else {}
    count = min(4, max((len(values) for values in columns.values()), default=0))
    return [
        {key: columns[key][i] if i < len(columns[key]) else "" for key in keys}
        for i in range(count)
    ]


def validate_profile(data):
    errors = {}
    numbers = {}
    for field, lower, upper in (
        ("income", 0, 1_000_000_000),
        ("expenses", 0, 1_000_000_000),
        ("goal", 0, 1_000_000_000_000),
        ("deduction_80c", 0, 150_000),
        ("deduction_nps", 0, 50_000),
        ("deduction_80d", 0, 50_000),
    ):
        raw = data.get(field, "")
        if field.startswith("deduction_") and raw == "":
            raw = "0"
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value < lower or value > upper or (field in ("income", "goal") and value == 0):
            errors[field] = f"Enter an amount from ₹{lower:,.0f} to ₹{upper:,.0f}."
        else:
            numbers[field] = value

    try:
        age = int(data.get("age", ""))
        if not 18 <= age <= 80:
            raise ValueError
    except (TypeError, ValueError):
        errors["age"] = "Age must be between 18 and 80."
        age = None

    if age is not None and numbers.get("deduction_80d", 0) > (50_000 if age >= 60 else 25_000):
        errors["deduction_80d"] = "This self/family 80D amount exceeds the age-based limit."

    if data.get("risk") not in ("Low", "Medium", "High"):
        errors["risk"] = "Choose a risk level."
    if data.get("dependents", "0") not in ("0", "1", "2", "3"):
        errors["dependents"] = "Choose a valid number of dependents."
    if data.get("has_insurance", "none") not in ("none", "health_only", "life_only", "both"):
        errors["has_insurance"] = "Choose a valid insurance option."

    return errors


def parse_plan_form(data):
    """Validate the extended planning inputs and return normalized values."""
    errors = validate_profile(data)
    values = {}
    specs = {
        "monthly_take_home": (0, 1_000_000_000, 0),
        "monthly_emi": (0, 1_000_000_000, 0),
        "debt_balance": (0, 1_000_000_000_000, 0),
        "debt_rate": (0, 100, 0),
        "liquid_savings": (0, 1_000_000_000_000, 0),
        "investments": (0, 1_000_000_000_000, 0),
        "life_cover": (0, 1_000_000_000_000, 0),
        "health_cover": (0, 1_000_000_000_000, 0),
        "goal_saved": (0, 1_000_000_000_000, 0),
        "goal_monthly": (0, 1_000_000_000, 0),
        "return_low": (0, 25, 4),
        "return_base": (0, 25, 8),
        "return_high": (0, 25, 12),
        "inflation": (0, 20, 6),
    }
    for key, (minimum, maximum, default) in specs.items():
        raw = data.get(key, "")
        try:
            value = float(raw) if raw not in ("", None) else default
        except (TypeError, ValueError):
            value = float("nan")
        if key == "monthly_take_home" and raw not in ("", None) and value == 0:
            errors[key] = "Enter take-home pay above zero, or leave this blank."
        elif not math.isfinite(value) or not minimum <= value <= maximum:
            errors[key] = f"Enter a value between {minimum:,} and {maximum:,}."
        else:
            values[key] = value

    for key, minimum, maximum, default in (
        ("goal_years", 1, 50, 10),
        ("emergency_months", 1, 24, 6),
    ):
        try:
            value = int(data.get(key) or default)
            if not minimum <= value <= maximum:
                raise ValueError
            values[key] = value
        except (TypeError, ValueError):
            errors[key] = f"Enter a whole number from {minimum} to {maximum}."

    if all(key in values for key in ("return_low", "return_base", "return_high")):
        if not values["return_low"] <= values["return_base"] <= values["return_high"]:
            errors["returns"] = "Scenario returns must increase from low to base to high."

    goal_name = (data.get("goal_name") or "Primary goal").strip()
    if not 1 <= len(goal_name) <= 60:
        errors["goal_name"] = "Goal name must be 1–60 characters."

    extra_keys = ("extra_goal_name", "extra_goal_amount", "extra_goal_years", "extra_goal_saved", "extra_goal_monthly")
    columns = [data.getlist(key) if hasattr(data, "getlist") else data.get(key, []) for key in extra_keys]
    columns = [column if isinstance(column, list) else [] for column in columns]
    if len({len(column) for column in columns}) > 1 or len(columns[0]) > 4:
        errors["extra_goals"] = "Enter no more than four complete extra goals."
    extras = []
    if "extra_goals" not in errors:
        for row in zip(*columns):
            if not any(str(value).strip() for value in row):
                continue
            name, target, years, saved, monthly = row
            try:
                name = str(name).strip()
                target = float(target)
                years = int(years)
                saved = float(saved or 0)
                monthly = float(monthly or 0)
                if (not 1 <= len(name) <= 60 or not math.isfinite(target) or not 0 < target <= 1_000_000_000_000
                    or not 1 <= years <= 50 or not math.isfinite(saved) or not 0 <= saved <= 1_000_000_000_000
                    or not math.isfinite(monthly) or not 0 <= monthly <= 1_000_000_000):
                    raise ValueError
                extras.append({"name": name, "amount": target, "years": years, "saved": saved, "monthly": monthly})
            except (TypeError, ValueError):
                errors["extra_goals"] = "Every extra goal needs a name, positive amount, and 1–50 year timeline."
                break

    for key in ("life_target", "health_target"):
        raw = data.get(key, "")
        if raw not in ("", None):
            try:
                target = float(raw)
                if not math.isfinite(target) or not 0 <= target <= 1_000_000_000_000:
                    raise ValueError
                values[key] = target
            except (TypeError, ValueError):
                errors[key] = "Enter a valid coverage target or leave this blank."

    if errors:
        return None, None, None, errors

    income = float(data["income"])
    take_home = values["monthly_take_home"] or income
    if take_home > income:
        return None, None, None, {"monthly_take_home": "Take-home pay cannot exceed the gross salary entered."}
    debt = values["debt_balance"]
    dependents = int(data.get("dependents", 0))
    profile = {
        "age": int(data["age"]),
        "income": income,
        "cash_income": take_home,
        "expenses": float(data["expenses"]),
        "goal_amount": float(data["goal"]),
        "risk": data["risk"],
        "dependents": dependents,
        "has_insurance": data.get("has_insurance", "none"),
        "deduction_80c": float(data.get("deduction_80c") or 0),
        "deduction_80d": float(data.get("deduction_80d") or 0),
        "deduction_nps": float(data.get("deduction_nps") or 0),
        **{key: values[key] for key in ("monthly_emi", "debt_balance", "debt_rate", "liquid_savings", "investments", "life_cover", "health_cover", "emergency_months")},
        "life_target": values.get("life_target", income * 12 * 10 + debt if dependents else 0),
        "health_target": values.get("health_target", 500_000 + dependents * 200_000),
    }
    goals = [{
        "name": goal_name,
        "amount": profile["goal_amount"],
        "years": values["goal_years"],
        "saved": values["goal_saved"],
        "monthly": values["goal_monthly"],
    }, *extras]
    assumptions = {key: values[key] for key in ("return_low", "return_base", "return_high", "inflation")}
    return profile, goals, assumptions, {}


def tax_estimate(income, age, deduction_80c=0, deduction_80d=0, deduction_nps=0):
    """Illustrative FY 2026-27 tax for a resident with ordinary salary only."""
    annual = income * 12
    if annual > 5_000_000:
        return None  # Surcharge and marginal relief need a fuller tax calculation.

    def slab_tax(taxable, slabs):
        total, floor = 0.0, 0.0
        for ceiling, rate in slabs:
            total += max(0, min(taxable, ceiling) - floor) * rate
            floor = ceiling
            if taxable <= ceiling:
                break
        return total

    new_slabs = [(400_000, 0), (800_000, .05), (1_200_000, .10),
                 (1_600_000, .15), (2_000_000, .20), (2_400_000, .25),
                 (float("inf"), .30)]
    old_exemption = 500_000 if age >= 80 else 300_000 if age >= 60 else 250_000
    old_slabs = [(old_exemption, 0), (500_000, .05), (1_000_000, .20),
                 (float("inf"), .30)]

    new_taxable = max(0, annual - min(annual, 75_000))
    old_base = max(0, annual - min(annual, 50_000))
    old_taxable = max(0, old_base - deduction_80c - deduction_80d - deduction_nps)

    def after_rebate(taxable, slabs, regime):
        tax = slab_tax(taxable, slabs)
        if regime == "new":
            if taxable <= 1_200_000:
                tax -= min(tax, 60_000)
            else:
                tax = min(tax, taxable - 1_200_000)  # Section 87A marginal relief
        elif taxable <= 500_000:
            tax -= min(tax, 12_500)
        return tax * 1.04  # 4% health and education cess

    new_tax = after_rebate(new_taxable, new_slabs, "new")
    old_tax_before = after_rebate(old_base, old_slabs, "old")
    old_tax_after = after_rebate(old_taxable, old_slabs, "old")

    return {
        "annual_income": round(annual),
        "new_taxable": round(new_taxable),
        "old_taxable": round(old_taxable),
        "new_regime_tax": round(new_tax),
        "old_regime_tax_before": round(old_tax_before),
        "old_regime_tax_after": round(old_tax_after),
        "tax_saved_deductions": round(old_tax_before - old_tax_after),
        "deduction_80c": round(deduction_80c),
        "deduction_80d": round(deduction_80d),
        "deduction_nps": round(deduction_nps),
        "better_regime": "New" if new_tax <= old_tax_after else "Old",
    }


def planning_summary(plan):
    lines = [
        f"Monthly take-home or entered cash income: ₹{plan['monthly_income']:,}; living expenses: ₹{plan['monthly_expenses']:,}; loan payments: ₹{plan['monthly_emi']:,}.",
        f"Monthly cash-flow surplus: ₹{plan['monthly_surplus']:,} (after living expenses and loan payments).",
        f"Emergency reserve target: ₹{plan['emergency_target']:,}; gap: ₹{plan['emergency_gap']:,}.",
        f"Net worth (assets minus debt): ₹{plan['net_worth']:,}.",
        f"Life-cover gap against the editable target: ₹{plan['life_gap']:,}; health-cover gap: ₹{plan['health_gap']:,}.",
        f"Debt payoff status at the entered payment and fixed rate: {plan['debt_payoff']['status']}"
        + (f", approximately {plan['debt_payoff']['months']} months." if plan['debt_payoff']['months'] is not None else "."),
        f"Total base-scenario monthly contributions needed for goals: ₹{plan['required_goal_monthly']:,}; current surplus: ₹{plan['monthly_surplus']:,}.",
        f"Assumptions: inflation {plan['assumptions']['inflation']}%; annual return scenarios {plan['assumptions']['return_low']}%, {plan['assumptions']['return_base']}%, {plan['assumptions']['return_high']}%. These are hypothetical returns, not forecasts.",
    ]
    for goal in plan["goals"]:
        base = goal["scenarios"]["base"]
        lines.append(
            f"Goal {goal['name']} in {goal['years']} years: today's amount ₹{goal['today_cost']:,}, "
            f"inflation-adjusted target ₹{goal['future_target']:,}, saved ₹{goal['saved']:,}, "
            f"planned monthly ₹{goal['planned_monthly']:,}, base-scenario monthly needed ₹{base['required_monthly']:,}, "
            f"projected shortfall ₹{base['shortfall']:,}."
        )
    return "\n".join(lines)


@app.route("/generate", methods=["POST"])
def generate():
    if not allow_anonymous_usage(app.config["DATABASE_PATH"], request.remote_addr or "unknown", "generate", 20):
        return render_template("index.html", errors={"service": "Plan limit reached. Try again in an hour."}, values=request.form, extra_rows=extra_goal_rows(request.form)), 429
    data = request.form

    profile, goals, assumptions, errors = parse_plan_form(data)
    if errors:
        return render_template("index.html", errors=errors, values=data, extra_rows=extra_goal_rows(data)), 400
    planning = build_plan(profile, goals, assumptions)
    primary_goal = {**goals[0], "future_target": planning["goals"][0]["future_target"]}

    state = {
        **profile,
        "primary_goal": primary_goal,
        "base_return": assumptions["return_base"],
        "planning_summary": planning_summary(planning),
        "chat_history": [],
        "ai_allowed": data.get("ai_consent") == "1",
        "user_query": "Generate a detailed financial plan"
    }

    try:
        result = smartfin.invoke(state)
    except Exception:
        logger.exception("Plan generation failed")
        return render_template("index.html", errors={"service": "The advisor is temporarily unavailable. Please try again."}, values=data, extra_rows=extra_goal_rows(data)), 503

    # Growth chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(result["growth"]) + 1), result["growth"], color="#2b6cb0", linewidth=2.5, label="Planned contributions")
    ax.fill_between(range(1, len(result["growth"]) + 1), result["growth"], alpha=0.1, color="#2b6cb0")
    ax.axhline(primary_goal["future_target"], color="#d69e2e", linestyle="--", linewidth=1.5, label="Inflation-adjusted target")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Primary Goal Scenario", fontsize=13, fontweight='bold', color="#1a365d", pad=12)
    ax.set_xlabel("Months", color="#4a5568")
    ax.set_ylabel("Value (₹)", color="#4a5568")
    ax.grid(True, alpha=0.4, linestyle='--', color="#e2e8f0")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7fafc")
    ax.tick_params(colors="#4a5568")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_edgecolor("#e2e8f0")

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # Tax estimate is shown only within its documented scope.
    tax = tax_estimate(
        profile["income"], profile["age"], profile["deduction_80c"],
        profile["deduction_80d"], profile["deduction_nps"]
    )

    token = str(uuid.uuid4())
    plan_data = {
        "response": result["response"],
        "graph": graph_url,
        "tax": tax,
        "profile": profile,
        "goals": goals,
        "assumptions": assumptions,
        "planning": planning,
        "ai_consent": state["ai_allowed"],
    }
    save_plan(app.config["DATABASE_PATH"], token, plan_data)
    previous_token = session.get("token")
    if previous_token:
        delete_plan(app.config["DATABASE_PATH"], previous_token)
    csrf = csrf_token()
    session.clear()
    session.permanent = True
    session["token"] = token
    session["csrf_token"] = csrf

    return redirect(url_for("result_page"))

@app.route("/result")
def result_page():
    token = session.get("token")
    d = load_plan(app.config["DATABASE_PATH"], token)
    if d is None:
        return redirect(url_for("home"))
    return render_template("result.html",
        response=d["response"],
        graph=d["graph"],
        tax=d["tax"],
        profile=d["profile"],
        planning=d["planning"],
        ai_enabled=llm is not None and d.get("ai_consent", False),
        ai_consented=d.get("ai_consent", False),
        chat_history=load_history(app.config["DATABASE_PATH"], token),
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        return jsonify({"error": "Send a question as JSON."}), 400
    query = data["query"].strip()
    if not query or len(query) > 2000:
        return jsonify({"error": "Question must be 1–2000 characters."}), 400

    token = session.get("token")
    plan_data = load_plan(app.config["DATABASE_PATH"], token)
    if plan_data is None:
        return jsonify({"error": "Generate a plan before asking a follow-up question."}), 401
    if not plan_data.get("ai_consent") or llm is None:
        return jsonify({"error": "Enable optional AI when creating the plan to use chat."}), 403
    if not allow_anonymous_usage(app.config["DATABASE_PATH"], request.remote_addr or "unknown", "chat", 30):
        return jsonify({"error": "Chat limit reached. Try again in an hour."}), 429
    profile = plan_data["profile"]
    planning = plan_data["planning"]

    state = {
        "age": profile["age"],
        "income": profile["income"],
        "expenses": profile["expenses"],
        "goal_amount": profile["goal_amount"],
        "risk": profile["risk"],
        "monthly_emi": profile["monthly_emi"],
        "cash_income": profile.get("cash_income", profile["income"]),
        "primary_goal": {**plan_data["goals"][0], "future_target": planning["goals"][0]["future_target"]},
        "base_return": plan_data["assumptions"]["return_base"],
        "planning_summary": planning_summary(planning),
        "chat_history": load_history(app.config["DATABASE_PATH"], token),
        "ai_allowed": True,
        "user_query": query,
    }

    try:
        result = smartfin.invoke(state)
    except Exception:
        logger.exception("Follow-up generation failed")
        return jsonify({"error": "The advisor is temporarily unavailable. Please try again."}), 503
    save_exchange(app.config["DATABASE_PATH"], token, query, result["response"])
    return jsonify({"response": result["response"]})


@app.route("/export")
def export_plan():
    token = session.get("token")
    data = load_plan(app.config["DATABASE_PATH"], token)
    if data is None:
        abort(404)
    response = jsonify({"plan": data, "chat_history": load_history(app.config["DATABASE_PATH"], token, limit=10000)})
    response.headers["Content-Disposition"] = 'attachment; filename="smartfin-plan.json"'
    return response


@app.route("/delete", methods=["POST"])
def delete_my_plan():
    token = session.get("token")
    if token:
        delete_plan(app.config["DATABASE_PATH"], token)
    session.pop("token", None)
    return redirect(url_for("plan"))

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
