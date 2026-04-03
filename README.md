#  SmartFin – Agentic AI Financial Advisor

SmartFin is an **agentic AI-powered financial advisor** built using **LangGraph, Groq LLMs, Streamlit, and online web search tools**.  
It provides **personalized financial planning**, **investment guidance**, and **real-time market-aware answers** through a conversational interface.

Unlike traditional chatbots, SmartFin uses a **deterministic agent architecture** with explicit reasoning steps, financial tools, and optional online research.

---

## 🚀 Live Application

🔗 **App Link:**  
👉 https://smartfin-agent-based-financial-advisor.onrender.com/ 


---

## 🧠 Key Features

- ✅ Personalized financial planning based on user profile  
- ✅ Goal-based investment timeline estimation  
- ✅ SIP growth projections with charts  
- ✅ Follow-up conversational chat (context-aware)  
- ✅ Agentic reasoning using LangGraph (not legacy agents)  
- ✅ Online market & stock research using web search  
- ✅ Safe & conservative financial advice (no guarantees)  

---

## 🏗️ Architecture Overview

SmartFin is built using a **state-driven agentic workflow** instead of a single monolithic LLM call.

---

## 🧩 Agentic Design (LangGraph)

The core intelligence is implemented using **LangGraph**, which allows explicit control over agent behavior.

### 🔸 State Object (`FinanceState`)
All agents operate on a shared state containing:
- User profile (age, income, expenses, risk)
- Financial calculations (monthly savings, goal timeline)
- Chat history
- Online search results
- Final AI response

---

### 🔸 Agent Nodes

#### 1️⃣ Profile Analyzer Node
- Computes monthly savings
- Normalizes user financial inputs

#### 2️⃣ Financial Calculator Node
- Estimates time to reach financial goals
- Generates SIP growth projections
- Ensures all numbers come from deterministic logic (no hallucination)

#### 3️⃣ Online Search Node
- Triggered only when user asks about:
  - Stocks
  - Markets
  - Investments
  - Financial news
- Uses Tavily search API for real-time information

#### 4️⃣ Advisor Node (LLM)
- Uses Groq-hosted `gpt-oss-20b` model
- Combines:
  - User profile
  - Calculated results
  - Search insights
  - Conversation history
- Produces structured, safe financial advice

---

## 🖥️ Frontend (Streamlit)

The Streamlit UI provides:
- Sidebar for user profile input
- Button to generate initial financial plan
- Interactive charts for investment growth
- Follow-up chat box for additional questions

---

## 📊 Visualizations

- 📈 SIP / Investment Growth Chart
- (Extensible to asset allocation pie charts, what-if simulations)

---

## 🔍 Online Search Integration

SmartFin can perform **real-time online research** when required:
- Uses Tavily Search API
- Searches are triggered intelligently based on query intent
- Results are summarized and grounded into the final response

> This enables questions like:
> - *“Which stocks should I invest in?”*
> - *“What are good long-term investments right now?”*

---

## 🛡️ Safety & Compliance

- ❌ No guaranteed returns
- ❌ No direct buy/sell recommendations
- ✅ Conservative assumptions
- ✅ Clear disclaimers
- ✅ Explainable reasoning

This makes SmartFin suitable for **FinTech demos, research, and production prototypes**.

---

## 🧪 Tech Stack

- **Frontend:** Streamlit  
- **LLM Inference:** Groq (`gpt-oss-120b`)  
- **Agent Framework:** LangGraph  
- **LLM Interface:** langchain-openai  
- **Search Tool:** Tavily API  
- **Visualization:** Matplotlib  
- **Environment Management:** python-dotenv  

---


