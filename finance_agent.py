import os
from io import StringIO
from typing import List, TypedDict

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LLM_NAME = "gpt-3.5-turbo"

# Initialize clients
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_NAME)
tavily = TavilyClient(api_key=TAVILY_API_KEY)
memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    """State definition for the LangGraph agent."""

    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str
    comparison: str
    feedback: str
    report: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    """Structured output model for search queries."""

    queries: List[str]


# Define the prompts for each node
GATHER_FINANCIALS_PROMPT = "You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."
ANALYZE_DATA_PROMPT = "You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."
RESEARCH_COMPETITORS_PROMPT = "You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = "You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."
WRITE_REPORT_PROMPT = "You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."
RESEARCH_CRITIQUE_PROMPT = "You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."


def gather_financials_node(state: AgentState) -> dict:
    """Reads financial CSV data and gathers insights."""
    df = pd.read_csv(StringIO(state["csv_file"]))
    financial_data_str = df.to_string(index=False)

    combined_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )
    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]
    response = model.invoke(messages)
    return {"financial_data": response.content}


def analyze_data_node(state: AgentState) -> dict:
    """Analyzes the gathered financial data."""
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}


def research_competitors_node(state: AgentState) -> dict:
    """Generates search queries for competitors and gathers data."""
    content = state.get("content", []) or []
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
                HumanMessage(content=competitor),
            ]
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}


def compare_performance_node(state: AgentState) -> dict:
    """Compares the target company performance with competitors."""
    content = "\n\n".join(state.get("content", []) or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def research_critique_node(state: AgentState) -> dict:
    """Researches additional information based on feedback."""
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state.get("content", []) or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def collect_feedback_node(state: AgentState) -> dict:
    """Collects critique and feedback on the comparison report."""
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}


def write_report_node(state: AgentState) -> dict:
    """Writes the final comprehensive financial report."""
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


def should_continue(state: AgentState) -> str:
    """Determines if the agent should continue refining the report."""
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"


def build_graph() -> StateGraph:
    """Compiles and returns the LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("gather_financials", gather_financials_node)
    builder.add_node("analyze_data", analyze_data_node)
    builder.add_node("research_competitors", research_competitors_node)
    builder.add_node("compare_performance", compare_performance_node)
    builder.add_node("collect_feedback", collect_feedback_node)
    builder.add_node("research_critique", research_critique_node)
    builder.add_node("write_report", write_report_node)

    builder.set_entry_point("gather_financials")

    builder.add_conditional_edges(
        "compare_performance",
        should_continue,
        {END: END, "collect_feedback": "collect_feedback"},
    )

    builder.add_edge("gather_financials", "analyze_data")
    builder.add_edge("analyze_data", "research_competitors")
    builder.add_edge("research_competitors", "compare_performance")
    builder.add_edge("collect_feedback", "research_critique")
    builder.add_edge("research_critique", "compare_performance")
    builder.add_edge("compare_performance", "write_report")

    return builder.compile(checkpointer=memory)


graph = build_graph()


def run_agent(
    task: str, competitors_text: str, max_revisions: int, csv_file_obj: bytes
):
    """Executes the agent workflow and yields updates for Gradio UI."""
    if csv_file_obj is None:
        yield "Please upload a CSV file.", ""
        return

    csv_data = csv_file_obj.decode("utf-8")
    competitors = [c.strip() for c in competitors_text.split("\n") if c.strip()]

    initial_state = {
        "task": task,
        "competitors": competitors,
        "csv_file": csv_data,
        "max_revisions": int(max_revisions),
        "revision_number": 1,
    }
    thread = {"configurable": {"thread_id": "1"}}

    output_log = "🚀 Initializing Agent...\n\n"
    final_report = "### ⏳ Generating final report...\n\nPlease wait. The agent is gathering financial data, searching for competitors, and running analysis. You can check the **Agent Progress Log** tab for live updates!"

    yield output_log, final_report

    for s in graph.stream(initial_state, thread):
        output_log += str(s) + "\n\n"

        for node, state_update in s.items():
            if isinstance(state_update, dict) and "report" in state_update:
                final_report = state_update["report"]

        yield output_log, final_report


def create_ui() -> gr.Blocks:
    """Builds and returns the Gradio UI."""
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #4f46e5, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    """

    with gr.Blocks(
        title="Finmetric Orchestrator AI Agent",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="pink",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=custom_css,
    ) as demo:
        gr.HTML(
            """
            <div class="main-header">Finmetric Orchestrator AI Agent</div>
            <div class="sub-header">AI-Powered Financial Analysis & Competitor Comparison</div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📊 Agent Configuration")
                    task_input = gr.Textbox(
                        label="Objective",
                        value="Analyze the financial performance of our company (NepalAICo.np) compared to competitors",
                        placeholder="Describe the main goal of the analysis...",
                    )
                    competitors_input = gr.Textbox(
                        label="Competitors (one per line)",
                        lines=3,
                        placeholder="Microsoft\nGoogle\nNvidia",
                    )
                    with gr.Row():
                        max_revisions_input = gr.Number(
                            label="Max Revisions", value=2, minimum=1
                        )

                    csv_upload = gr.File(
                        label="Upload Company Financials (CSV)",
                        file_types=[".csv"],
                        type="binary",
                    )
                    start_button = gr.Button("🚀 Start Analysis", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📋 Final Report"):
                        final_report_box = gr.Markdown(label="Generated Insights")
                    with gr.TabItem("⚙️ Agent Progress Log"):
                        output_log_box = gr.Textbox(
                            label="Live Trace", lines=20, max_lines=25, show_label=False
                        )

        start_button.click(
            fn=run_agent,
            inputs=[task_input, competitors_input, max_revisions_input, csv_upload],
            outputs=[output_log_box, final_report_box],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
