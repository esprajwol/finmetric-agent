# Finmetric Orchestrator AI Agent

Finmetric Orchestrator is an AI-powered financial analysis and competitor comparison agent. It leverages LangGraph and OpenAI to analyze company financial data (via CSV) and research competitors on the web (via Tavily) to generate comprehensive comparative reports.

## Prerequisites

- Python 3.9+
- An OpenAI API Key
- A Tavily API Key

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:esprajwol/finmetric-agent.git
   cd sim-agent
   ```

2. **Install the dependencies:**
   Make sure to install the exact required dependencies provided in the `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   You must set up your API keys before running the agent. 
   
   Copy the provided `.env.example` file to create a new `.env` file:
   ```bash
   cp .env.example .env
   ```
   
   Open the new `.env` file and fill in your keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # Optional: LangSmith Tracing configuration
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   LANGCHAIN_TRACING_V2=false
   LANGCHAIN_PROJECT=sim-agent
   ```

## Running the Application

This application uses **Gradio** for its user interface.

To start the server with **hot-reloading enabled** (the app automatically refreshes when you save code changes), use the Gradio CLI:

```bash
gradio finance_agent.py
```

*Alternatively, you can run it as a standard python script:*
```bash
python finance_agent.py
```

The terminal will provide a local URL (usually `http://127.0.0.1:7860`). Open that link in your browser to interact with the Finmetric Orchestrator!
