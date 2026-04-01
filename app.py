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
app.secret_key = secrets.token_hex(16)

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
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.form

    session["profile"] = {
        "age": int(data["age"]),
        "income": float(data["income"]),
        "expenses": float(data["expenses"]),
        "goal_amount": float(data["goal"]),
        "risk": data["risk"],
    }

    state = {
        **session["profile"],
        "chat_history": [],
        "user_query": "Generate a detailed financial plan"
    }

    result = smartfin.invoke(state)

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

    # Store large data server-side; only a small token goes in the cookie
    token = str(uuid.uuid4())
    _result_store[token] = {
        "response": result["response"],
        "graph": graph_url,
        "profile": session["profile"],
    }
    session["token"] = token

    return redirect(url_for("result_page"))

@app.route("/result")
def result_page():
    token = session.get("token")
    if not token or token not in _result_store:
        return redirect(url_for("home"))
    data = _result_store[token]
    return render_template("result.html",
        response=data["response"],
        graph=data["graph"],
        profile=data["profile"]
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