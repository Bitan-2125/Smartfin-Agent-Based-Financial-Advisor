import os
import uuid
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import — avoids Tkinter thread issues
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import io
import base64
import secrets

from dotenv import load_dotenv
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

# -----------------------
# ENV
# -----------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

# Server-side store to avoid oversized session cookies
_result_store = {}

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3,
)

search_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="finance",
)

SYSTEM_PROMPT = """
You are SmartFin, a professional AI financial advisor specializing in the Indian market.

Rules:
- All amounts are in Indian Rupees (₹). Never use $ or USD.
- Recommend Indian investment instruments: PPF, EPF, NPS, ELSS, Mutual Funds (SIP), FD, RD, Sovereign Gold Bonds, NSC, Sukanya Samriddhi, REITs, direct equity (NSE/BSE).
- Reference Indian tax laws: Section 80C, 80D, 80CCD(1B), LTCG, STCG, new vs old tax regime.
- Use Indian market return benchmarks: Nifty 50 (~12% long-term), debt funds (~6-7%), FD (~7%), PPF (~7.1%).
- Never promise guaranteed returns
- Be conservative and realistic
- Explain assumptions clearly
- End with follow-up questions
"""

# -----------------------
# LOGIC (same as yours)
# -----------------------
def monthly_savings(income, expenses):
    return max(income - expenses, 0)

def goal_timeline(goal, monthly, annual_return=0.10):
    r = annual_return / 12
    bal, months = 0, 0
    while bal < goal and months < 600:
        bal = bal * (1 + r) + monthly
        months += 1
    return months

def sip_projection(monthly, months, annual_return=0.10):
    r = annual_return / 12
    bal = 0
    values = []
    for _ in range(months):
        bal = bal * (1 + r) + monthly
        values.append(bal)
    return values

class FinanceState(TypedDict):
    age: int
    income: float
    expenses: float
    goal_amount: float
    risk: str
    monthly_saving: float
    months_to_goal: int
    growth: list
    chat_history: list
    user_query: str
    response: str
    search_result: str

def analyze(state):
    state["monthly_saving"] = monthly_savings(
        state["income"], state["expenses"]
    )
    return state

def calculate(state):
    state["months_to_goal"] = goal_timeline(
        state["goal_amount"], state["monthly_saving"]
    )
    state["growth"] = sip_projection(
        state["monthly_saving"],
        min(state["months_to_goal"], 360)
    )
    return state

def search(state):
    q = state.get("user_query", "").lower()
    if any(k in q for k in ["stock", "market", "invest", "nifty", "sensex", "mutual fund", "sip", "nse", "bse"]):
        result = search_tool.run(state["user_query"])
        summaries = [f"{r['title']}: {r['content']}" for r in result["results"]]
        state["search_result"] = "\n".join(summaries)
    else:
        state["search_result"] = ""
    return state

def advisor(state):
    prompt = f"""
User Profile (Indian investor):
Age: {state['age']}
Monthly Savings: ₹{state['monthly_saving']:,.0f}
Financial Goal: ₹{state['goal_amount']:,.0f}
Risk Tolerance: {state['risk']}

{state.get('search_result','')}
User Query: {state.get('user_query','')}

Provide advice specific to the Indian market. Use ₹ for all amounts.
"""
    res = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    state["response"] = res.content
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
    return render_template("landing.html")

@app.route("/plan")
def plan():
    return render_template("index.html", errors={}, values={})

# -----------------------
# HELPERS
# -----------------------
def validate_profile(data):
    errors = {}
    try:
        age = int(data.get("age", 0))
        if not (18 <= age <= 80):
            errors["age"] = "Age must be between 18 and 80."
    except ValueError:
        errors["age"] = "Enter a valid age."

    try:
        income = float(data.get("income", 0))
        if income <= 0:
            errors["income"] = "Income must be greater than 0."
    except ValueError:
        errors["income"] = "Enter a valid income."

    try:
        expenses = float(data.get("expenses", 0))
        if expenses < 0:
            errors["expenses"] = "Expenses cannot be negative."
    except ValueError:
        errors["expenses"] = "Enter valid expenses."

    if "income" not in errors and "expenses" not in errors:
        if float(data.get("expenses", 0)) >= float(data.get("income", 0)):
            errors["expenses"] = "Expenses must be less than income to have savings."

    try:
        goal = float(data.get("goal", 0))
        if goal <= 0:
            errors["goal"] = "Goal amount must be greater than 0."
    except ValueError:
        errors["goal"] = "Enter a valid goal amount."

    return errors


def allocation_for_risk(risk):
    """Return portfolio allocation percentages based on risk profile."""
    if risk == "Low":
        return {"PPF/FD/Debt": 45, "ELSS/Equity MF": 20, "NPS": 15, "Gold/SGB": 10, "Emergency Fund": 10}
    elif risk == "Medium":
        return {"ELSS/Equity MF": 40, "PPF/FD/Debt": 25, "NPS": 15, "Gold/SGB": 10, "Emergency Fund": 10}
    else:  # High
        return {"Direct Equity/MF": 55, "ELSS": 20, "NPS": 10, "Gold/SGB": 10, "Emergency Fund": 5}


def tax_estimate(income, risk):
    """Compute basic tax estimates: old vs new regime, 80C savings."""
    annual = income * 12

    # New regime slabs (FY 2024-25)
    def new_regime_tax(inc):
        slabs = [(300000, 0), (300000, 0.05), (300000, 0.10),
                 (300000, 0.15), (300000, 0.20), (float('inf'), 0.30)]
        tax, remaining = 0, inc
        for slab, rate in slabs:
            taxable = min(remaining, slab)
            tax += taxable * rate
            remaining -= taxable
            if remaining <= 0:
                break
        return max(0, tax - 25000)  # standard rebate u/s 87A up to ₹25k

    # Old regime slabs
    def old_regime_tax(inc):
        slabs = [(250000, 0), (250000, 0.05), (500000, 0.20), (float('inf'), 0.30)]
        tax, remaining = 0, inc
        for slab, rate in slabs:
            taxable = min(remaining, slab)
            tax += taxable * rate
            remaining -= taxable
            if remaining <= 0:
                break
        return max(0, tax - 12500)  # rebate u/s 87A

    deduction_80c = min(150000, annual * 0.20)   # assume 20% of income goes to 80C
    deduction_80d = 25000                          # standard health insurance
    deduction_nps = 50000                          # 80CCD(1B)
    total_deductions = deduction_80c + deduction_80d + deduction_nps

    new_tax = new_regime_tax(annual)
    old_tax_before = old_regime_tax(annual)
    old_tax_after  = old_regime_tax(max(0, annual - total_deductions))
    tax_saved = old_tax_before - old_tax_after

    return {
        "annual_income": annual,
        "new_regime_tax": round(new_tax),
        "old_regime_tax_before": round(old_tax_before),
        "old_regime_tax_after": round(old_tax_after),
        "tax_saved_deductions": round(tax_saved),
        "deduction_80c": round(deduction_80c),
        "deduction_80d": deduction_80d,
        "deduction_nps": deduction_nps,
        "better_regime": "New" if new_tax < old_tax_after else "Old",
    }


def make_allocation_chart(allocation):
    """Render portfolio allocation pie chart, return base64 PNG."""
    labels = list(allocation.keys())
    sizes  = list(allocation.values())
    colors = ["#2b6cb0", "#38a169", "#d69e2e", "#e53e3e", "#805ad5"]

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9, "color": "#2d3748"}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8)

    ax.set_title("Recommended Portfolio Allocation", fontsize=12,
                 fontweight="bold", color="#1a365d", pad=14)
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def insurance_guidance(income, age, dependents, has_insurance):
    """
    Compute insurance recommendations using Indian market rules of thumb.
    - Life cover: 10-15x annual income (higher multiplier for more dependents / younger age)
    - Health cover: ₹5L base, +₹2L per dependent, +₹5L if age > 45
    - Term plan premium estimate: ~0.3-0.5% of sum assured per year
    """
    annual = income * 12
    dep = int(dependents)

    # Life cover multiplier
    if dep == 0:
        multiplier = 10
    elif dep <= 2:
        multiplier = 12
    else:
        multiplier = 15

    # Reduce multiplier if already has life cover
    if has_insurance in ("life_only", "both"):
        multiplier = max(0, multiplier - 5)

    recommended_life = annual * multiplier

    # Health cover
    base_health = 500000
    health_cover = base_health + (dep * 200000)
    if age > 45:
        health_cover += 500000
    if has_insurance in ("health_only", "both"):
        health_cover = max(0, health_cover - 500000)  # already have base

    # Term plan annual premium estimate (~0.4% of SA for 30s, ~0.6% for 40s)
    premium_rate = 0.004 if age < 40 else 0.006
    term_premium = recommended_life * premium_rate if recommended_life > 0 else 0

    gaps = []
    if has_insurance == "none":
        gaps.append("No life or health cover — high financial risk for dependents")
    if has_insurance == "health_only":
        gaps.append("No life/term cover — income replacement risk")
    if has_insurance == "life_only":
        gaps.append("No health cover — medical costs can erode savings quickly")
    if dep > 0 and has_insurance in ("none", "health_only"):
        gaps.append(f"You have {dep} dependent(s) with no life cover")

    return {
        "recommended_life_cover": round(recommended_life),
        "recommended_health_cover": round(health_cover),
        "estimated_term_premium": round(term_premium),
        "gaps": gaps,
        "has_insurance": has_insurance,
        "dependents": dep,
        "needs_life": has_insurance in ("none", "health_only"),
        "needs_health": has_insurance in ("none", "life_only"),
    }



    """Render portfolio allocation pie chart, return base64 PNG."""
    labels = list(allocation.keys())
    sizes  = list(allocation.values())
    colors = ["#2b6cb0", "#38a169", "#d69e2e", "#e53e3e", "#805ad5"]

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9, "color": "#2d3748"}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8)

    ax.set_title("Recommended Portfolio Allocation", fontsize=12,
                 fontweight="bold", color="#1a365d", pad=14)
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/generate", methods=["POST"])
def generate():
    data = request.form

    # --- Input validation ---
    errors = validate_profile(data)
    if errors:
        return render_template("index.html", errors=errors, values=data)

    profile = {
        "age": int(data["age"]),
        "income": float(data["income"]),
        "expenses": float(data["expenses"]),
        "goal_amount": float(data["goal"]),
        "risk": data["risk"],
        "dependents": int(data.get("dependents", 0)),
        "has_insurance": data.get("has_insurance", "none"),
    }
    session["profile"] = profile

    state = {
        **profile,
        "chat_history": [],
        "user_query": "Generate a detailed financial plan"
    }

    result = smartfin.invoke(state)

    # Growth chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result["growth"], color="#2b6cb0", linewidth=2.5)
    ax.fill_between(range(len(result["growth"])), result["growth"], alpha=0.1, color="#2b6cb0")
    ax.set_title("Portfolio Growth Projection", fontsize=13, fontweight='bold', color="#1a365d", pad=12)
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

    # Allocation chart + tax estimate + insurance
    allocation = allocation_for_risk(profile["risk"])
    alloc_chart = make_allocation_chart(allocation)
    tax = tax_estimate(profile["income"], profile["risk"])
    insurance = insurance_guidance(
        profile["income"], profile["age"],
        profile["dependents"], profile["has_insurance"]
    )

    token = str(uuid.uuid4())
    _result_store[token] = {
        "response": result["response"],
        "graph": graph_url,
        "alloc_chart": alloc_chart,
        "allocation": allocation,
        "tax": tax,
        "insurance": insurance,
        "profile": profile,
    }
    session["token"] = token

    return redirect(url_for("result_page"))

@app.route("/result")
def result_page():
    token = session.get("token")
    if not token or token not in _result_store:
        return redirect(url_for("home"))
    d = _result_store[token]
    return render_template("result.html",
        response=d["response"],
        graph=d["graph"],
        alloc_chart=d["alloc_chart"],
        allocation=d["allocation"],
        tax=d["tax"],
        insurance=d["insurance"],
        profile=d["profile"]
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    token = session.get("token")
    profile = _result_store.get(token, {}).get("profile", {}) if token else {}

    state = {
        "age": profile.get("age", 25),
        "income": profile.get("income", 0),
        "expenses": profile.get("expenses", 0),
        "goal_amount": profile.get("goal_amount", 0),
        "risk": profile.get("risk", "Low"),
        "chat_history": [],
        "user_query": data["query"]
    }

    result = smartfin.invoke(state)
    return jsonify({"response": result["response"]})

if __name__ == "__main__":
    app.run(debug=True)