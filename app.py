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
# ENV + LLM + Search SETUP
# -------------------------------
load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3,
)

tavily_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="finance",
    search_depth="basic",
)

SYSTEM_PROMPT = """
You are SmartFin, a professional AI financial advisor.

Rules:
- Never promise guaranteed returns
- Be conservative and realistic
- Use online search results when relevant
- Answer clearly and helpfully
"""

# -------------------------------
# FINANCIAL CALC TOOLS
# -------------------------------
def monthly_savings(income, expenses):
    return income - expenses

def goal_timeline(goal_amount, monthly_saving, annual_return=0.10):
    r = annual_return / 12
    balance = 0
    months = 0
    while balance < goal_amount and months < 600:
        balance = balance * (1 + r) + monthly_saving
        months += 1
    return months

def sip_projection(monthly, months, annual_return=0.10):
    r = annual_return / 12
    values = []
    balance = 0
    for _ in range(months):
        balance = balance * (1 + r) + monthly
        values.append(balance)
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
    response: str
    growth: list
    chat_history: list
    user_query: str
    search_result: str

# -------------------------------
# LANGGRAPH NODES
# -------------------------------
def analyze_profile(state: FinanceState):
    state["monthly_saving"] = monthly_savings(state["income"], state["expenses"])
    return state

def calculate_plan(state: FinanceState):
    state["months_to_goal"] = goal_timeline(
        state["goal_amount"], state["monthly_saving"]
    )
    state["growth"] = sip_projection(
        state["monthly_saving"], min(state["months_to_goal"], 360)
    )
    return state

def search_node(state: FinanceState):
    qry = state.get("user_query", "")
    if any(k in qry.lower() for k in ["stock", "invest", "market", "share", "news"]):
        result = tavily_tool.run(qry)
        # build a simple text summary of the search results
        snippets = []
        for item in result["results"]:
            title = item.get("title", "")
            content = item.get("content", "")
            snippets.append(f"**{title}**: {content}")
        state["search_result"] = "\n\n".join(snippets)
    else:
        state["search_result"] = ""
    return state

def advisor_node(state: FinanceState):
    context = "\n".join(state.get("chat_history", []))
    prompt = f"""
Conversation:
{context}

User Profile:
Age: {state['age']}
Monthly Savings: {state['monthly_saving']}
Goal: {state['goal_amount']}
Risk: {state['risk']}

Online Search (if any):
{state.get('search_result', '')}

User Query:
{state.get('user_query', '')}

Answer helpfully with clear reasoning.
"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    res = llm.invoke(messages)
    state["response"] = res.content
    # store messages into history
    state.setdefault("chat_history", []).append(f"User: {state.get('user_query','')}")
    state["chat_history"].append(f"AI: {res.content}")
    return state

# -------------------------------
# BUILD LANGGRAPH
# -------------------------------
graph = StateGraph(FinanceState)

graph.add_node("analyze", analyze_profile)
graph.add_node("calculate", calculate_plan)
graph.add_node("search", search_node)
graph.add_node("advisor", advisor_node)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "calculate")
graph.add_edge("calculate", "search")
graph.add_edge("search", "advisor")
graph.add_edge("advisor", END)

smartfin_graph = graph.compile()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="SmartFin AI", layout="wide")
st.title("💸 SmartFin – Conversational Advisor with Web Search")

with st.sidebar:
    st.header("📋 Your Profile")
    age = st.number_input("Age", 18, 65, value=22)
    income = st.number_input("Monthly Income", value=50000)
    expenses = st.number_input("Monthly Expenses", value=30000)
    goal_amount = st.number_input("Goal Amount", value=500000)
    risk = st.selectbox("Risk Appetite", ["Low", "Medium", "High"])

# INITIAL PLAN
if st.button("Generate Financial Plan"):
    with st.spinner("Calculating..."):
        state = {
            "age": age,
            "income": income,
            "expenses": expenses,
            "goal_amount": goal_amount,
            "risk": risk,
            "chat_history": [],
            "user_query": "Generate plan"
        }
        result = smartfin_graph.invoke(state)

    st.subheader("📊 Financial Plan")
    st.markdown(result["response"])

    st.subheader("📈 Growth Projection")
    fig, ax = plt.subplots()
    ax.plot(result["growth"])
    ax.set_xlabel("Months")
    ax.set_ylabel("Projected Value")
    st.pyplot(fig)

    st.session_state["chat_history"] = result["chat_history"]

# FOLLOW-UP CHAT
st.subheader("💬 Ask a Follow-Up Question")
followup = st.text_input("Ask about investing, stocks, goals...")

if st.button("Ask SmartFin"):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    state = {
        "age": age,
        "income": income,
        "expenses": expenses,
        "goal_amount": goal_amount,
        "risk": risk,
        "chat_history": st.session_state["chat_history"],
        "user_query": followup
    }

    with st.spinner("Thinking..."):
        result = smartfin_graph.invoke(state)

    st.session_state["chat_history"] = result["chat_history"]
    st.markdown(result["response"])
