"""Plan-and-execute research agent with explicit reliability contracts."""

from __future__ import annotations

import time
from typing import Annotated, Literal, NotRequired, TypedDict

import groq
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.contracts import parse_tool_result

MAX_TOOL_ROUNDS = 4
MAX_TOOL_ERRORS = 2
RATE_LIMIT_RETRY_DELAY = 5


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    revision_count: int
    run_status: NotRequired[str]
    critic_status: NotRequired[str]
    citation_status: NotRequired[str]


def _is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "rate" in text


def _tool_rounds(messages: list[AnyMessage]) -> int:
    return sum(1 for message in messages if message.type == "tool")


def _recent_tool_errors(messages: list[AnyMessage]) -> int:
    count = 0
    for message in reversed(messages):
        if message.type != "tool":
            if message.type == "ai":
                continue
            break
        result = parse_tool_result(str(message.content))
        if result is not None:
            if result.ok:
                break
        elif "error" not in str(message.content).lower():
            break
        count += 1
    return count


def _citation_status(messages: list[AnyMessage], draft: str) -> str:
    evidence_ids: list[str] = []
    for message in messages:
        if message.type != "tool":
            continue
        result = parse_tool_result(str(message.content))
        if result:
            evidence_ids.extend(item.evidence_id for item in result.evidence)
    if not evidence_ids:
        return "not_required"
    if any(evidence_id in draft for evidence_id in evidence_ids) or "source" in draft.lower():
        return "present"
    return "missing"


def create_agent_graph(
    llm=None,
    tools=None,
    rate_limit_delay: float = RATE_LIMIT_RETRY_DELAY,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    max_tool_errors: int = MAX_TOOL_ERRORS,
):
    """Build the production graph, accepting injected dependencies for tests."""

    llm = llm or ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    if tools is None:
        from src.tools import get_tools

        tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def planner_node(state: AgentState):
        query = state["messages"][0].content if state["messages"] else ""
        sys_msg = SystemMessage(
            content=(
                "You are a planning assistant. Break the user's research question "
                "into concrete sub-tasks. Use 1 task for simple questions and 2-4 "
                "for complex questions. Output only a brief numbered list."
            )
        )
        try:
            response = llm.invoke([sys_msg, HumanMessage(content=query)])
        except Exception as error:
            if not _is_rate_limit(error):
                return {
                    "run_status": "planner_failed",
                    "messages": [AIMessage(content="The research plan could not be created.")],
                }
            time.sleep(rate_limit_delay)
            try:
                response = llm.invoke([sys_msg, HumanMessage(content=query)])
            except Exception as retry_error:
                return {
                    "run_status": "planner_failed",
                    "messages": [
                        AIMessage(content=f"Planning failed after retry: {type(retry_error).__name__}")
                    ],
                }
        return {"plan": response.content, "revision_count": 0, "run_status": "planned"}

    def executor_node(state: AgentState):
        sys_msg = SystemMessage(
            content=(
                "You are the research executor. Follow this plan:\n"
                f"{state.get('plan', '')}\n\n"
                "Use tools when evidence is needed. Synthesize a cohesive draft after "
                "the plan is complete. Cite structured evidence IDs such as [KB-1] "
                "or [CALC-1] when tool evidence supports a claim."
            )
        )
        try:
            response = llm_with_tools.invoke([sys_msg] + state["messages"])
        except groq.BadRequestError:
            retry_msg = HumanMessage(
                content=(
                    "The previous tool call was malformed. Retry with concise arguments "
                    "and valid JSON, or provide a final answer without that tool."
                )
            )
            try:
                response = llm_with_tools.invoke([sys_msg] + state["messages"] + [retry_msg])
            except Exception as error:
                return {
                    "run_status": "executor_failed",
                    "messages": [AIMessage(content=f"Tool-call formatting failed: {type(error).__name__}")],
                }
        except Exception as error:
            if not _is_rate_limit(error):
                return {
                    "run_status": "executor_failed",
                    "messages": [AIMessage(content=f"Executor failed: {type(error).__name__}")],
                }
            time.sleep(rate_limit_delay)
            try:
                response = llm_with_tools.invoke([sys_msg] + state["messages"])
            except Exception as retry_error:
                return {
                    "run_status": "executor_failed",
                    "messages": [AIMessage(content=f"Executor retry failed: {type(retry_error).__name__}")],
                }
        return {"messages": [response], "run_status": "executing"}

    def synthesis_node(state: AgentState):
        prompt = SystemMessage(
            content=(
                "Produce the best final answer from the available evidence. Do not "
                "request more tools. State limitations clearly and cite evidence IDs "
                "when present. This is a forced synthesis because the tool budget or "
                "tool-error budget was reached."
            )
        )
        try:
            response = llm.invoke([prompt] + state["messages"])
        except Exception as error:
            return {
                "run_status": "synthesis_failed",
                "messages": [AIMessage(content="The agent could not synthesize a final answer.")],
            }
        return {
            "run_status": "synthesized_with_limit",
            "messages": [response],
            "citation_status": _citation_status(state["messages"], str(response.content)),
        }

    def should_continue_executor(state: AgentState) -> Literal["tools", "critic", "synthesis", END]:
        if state.get("run_status") == "executor_failed":
            return END
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return "critic"
        if _tool_rounds(state["messages"]) >= max_tool_rounds:
            return "synthesis"
        if _recent_tool_errors(state["messages"]) >= max_tool_errors:
            return "synthesis"
        return "tools"

    def critic_node(state: AgentState):
        messages = state["messages"]
        draft = str(messages[-1].content)
        citation_status = _citation_status(messages, draft)
        if citation_status == "missing" and state.get("revision_count", 0) == 0:
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            "Critic Feedback: cite the supporting evidence IDs in the "
                            "draft, or explicitly state that the claim is unsupported."
                        )
                    )
                ],
                "revision_count": 1,
                "critic_status": "revision_requested",
                "citation_status": citation_status,
            }

        prompt = HumanMessage(
            content=(
                f"Original query: {messages[0].content}\n"
                f"Plan: {state.get('plan', '')}\n"
                f"Draft: {draft}\n"
                f"Citation status: {citation_status}\n"
                "Reply APPROVED only if the draft is complete and evidence-supported. "
                "Otherwise provide one brief correction."
            )
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a strict research reviewer. Check correctness, "
                            "completeness, and support for important claims."
                        )
                    ),
                    prompt,
                ]
            )
        except Exception as error:
            if _is_rate_limit(error):
                time.sleep(rate_limit_delay)
                try:
                    response = llm.invoke([prompt])
                except Exception:
                    return {
                        "run_status": "critic_unavailable",
                        "critic_status": "unavailable",
                        "citation_status": citation_status,
                    }
            else:
                return {
                    "run_status": "critic_unavailable",
                    "critic_status": "unavailable",
                    "citation_status": citation_status,
                }

        review = str(response.content).strip()
        if "APPROVED" in review.upper() or state.get("revision_count", 0) >= 1:
            return {
                "run_status": (
                    "completed_after_revision"
                    if state.get("revision_count", 0)
                    else "completed"
                ),
                "critic_status": "approved",
                "citation_status": citation_status,
            }
        return {
            "messages": [HumanMessage(content=f"Critic Feedback: {review}")],
            "revision_count": state.get("revision_count", 0) + 1,
            "critic_status": "revision_requested",
            "citation_status": citation_status,
        }

    def should_loop_critic(state: AgentState) -> Literal["executor", END]:
        if state["messages"][-1].type == "human" and "Critic Feedback:" in str(
            state["messages"][-1].content
        ):
            return "executor"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("critic", critic_node)
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        lambda state: END if state.get("run_status") == "planner_failed" else "executor",
    )
    workflow.add_conditional_edges("executor", should_continue_executor)
    workflow.add_edge("tools", "executor")
    workflow.add_edge("synthesis", "critic")
    workflow.add_conditional_edges("critic", should_loop_critic)
    return workflow.compile()
