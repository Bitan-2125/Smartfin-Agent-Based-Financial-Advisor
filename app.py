# ===============================
# SMARTFIN AI – main.py
# ===============================

import os
import streamlit as st
import matplotlib.pyplot as plt
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

# -------------------------------
# ENV SETUP
# -------------------------------
load_dotenv()

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
You are SmartFin, a professional AI financial advisor.

Rules:
- Never promise guaranteed returns
- Be conservative and realistic
- Explain reasoning clearly
- Use web search when relevant
"""

# -------------------------------
# FINANCE UTILS
# -------------------------------
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

# -------------------------------
# LANGGRAPH STATE
# -------------------------------
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

# -------------------------------
# LANGGRAPH NODES
# -------------------------------
def analyze(state: FinanceState):
    state["monthly_saving"] = monthly_savings(
        state["income"], state["expenses"]
    )
    return state

def calculate(state: FinanceState):
    state["months_to_goal"] = goal_timeline(
        state["goal_amount"], state["monthly_saving"]
    )
    state["growth"] = sip_projection(
        state["monthly_saving"],
        min(state["months_to_goal"], 360)
    )
    return state

def search(state: FinanceState):
    q = state.get("user_query", "").lower()
    if any(k in q for k in ["stock", "market", "invest", "share", "news"]):
        result = search_tool.run(state["user_query"])
        summaries = []
        for r in result["results"]:
            summaries.append(f"- {r['title']}: {r['content']}")
        state["search_result"] = "\n".join(summaries)
    else:
        state["search_result"] = ""
    return state

def advisor(state: FinanceState):
    prompt = f"""
User Profile:
Age: {state['age']}
Monthly Savings: {state['monthly_saving']}
Goal Amount: {state['goal_amount']}
Risk Appetite: {state['risk']}

Web Research:
{state.get('search_result', '')}

Conversation History:
{state.get('chat_history', [])}

User Query:
{state.get('user_query', '')}

Give a clear, conservative financial response.
"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    res = llm.invoke(messages)
    state["response"] = res.content
    state.setdefault("chat_history", []).append(f"User: {state['user_query']}")
    state["chat_history"].append(f"AI: {res.content}")
    return state

# -------------------------------
# GRAPH BUILD
# -------------------------------
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

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="SmartFin AI",
    page_icon="💸",
    layout="wide"
)

# -------------------------------
# PREMIUM UI CSS
# -------------------------------
st.markdown("""
<style>
.hero {
  background: linear-gradient(135deg, #6C63FF, #8A85FF);
  padding: 80px 50px;
  border-radius: 24px;
  color: white;
  text-align: center;
}
.hero h1 { font-size: 56px; font-weight: 800; }
.hero p { font-size: 20px; opacity: 0.95; }

.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 28px;
  margin-top: 60px;
}
.feature-card {
  background: rgba(255,255,255,0.08);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.15);
  padding: 28px;
  border-radius: 18px;
  transition: 0.3s;
}
.feature-card:hover { transform: translateY(-6px); }
.feature-icon { font-size: 36px; margin-bottom: 12px; }

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
  gap: 30px;
  margin-top: 60px;
  padding-top: 40px;
  border-top: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("📋 Profile")
    age = st.number_input("Age", 18, 65, 22)
    income = st.number_input("Monthly Income", value=50000)
    expenses = st.number_input("Monthly Expenses", value=30000)
    goal_amount = st.number_input("Goal Amount", value=500000)
    risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"])

# -------------------------------
# LANDING
# -------------------------------
st.markdown("""
<div class="hero">
  <h1>SmartFin AI</h1>
  <p>Your AI-Powered Financial Advisor</p>
</div>

<div class="feature-grid">
  <div class="feature-card"><div class="feature-icon">🎯</div><h3>Goal Planning</h3><p>Personalized financial goals.</p></div>
  <div class="feature-card"><div class="feature-icon">📈</div><h3>Projections</h3><p>Realistic long-term growth.</p></div>
  <div class="feature-card"><div class="feature-icon">🔍</div><h3>Live Research</h3><p>Market insights via web.</p></div>
  <div class="feature-card"><div class="feature-icon">🛡️</div><h3>Risk Aware</h3><p>Conservative advice.</p></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# PLAN GENERATION
# -------------------------------
st.divider()
if st.button("🚀 Generate Financial Plan", use_container_width=True):
    with st.spinner("Analyzing your finances..."):
        state = {
            "age": age,
            "income": income,
            "expenses": expenses,
            "goal_amount": goal_amount,
            "risk": risk,
            "chat_history": [],
            "user_query": "Generate a financial plan"
        }
        result = smartfin.invoke(state)

    st.subheader("📊 Financial Plan")
    st.markdown(result["response"])

    st.subheader("📈 Growth Projection")
    fig, ax = plt.subplots()
    ax.plot(result["growth"])
    ax.set_xlabel("Months")
    ax.set_ylabel("Portfolio Value")
    st.pyplot(fig)

    st.session_state["chat_history"] = result["chat_history"]

# -------------------------------
# CHAT
# -------------------------------
st.subheader("💬 Ask SmartFin")
q = st.text_input("Ask about investments, stocks, goals...")

if st.button("Ask"):
    state = {
        "age": age,
        "income": income,
        "expenses": expenses,
        "goal_amount": goal_amount,
        "risk": risk,
        "chat_history": st.session_state.get("chat_history", []),
        "user_query": q
    }
    with st.spinner("Thinking..."):
        result = smartfin.invoke(state)

    st.session_state["chat_history"] = result["chat_history"]
    st.markdown(result["response"])
