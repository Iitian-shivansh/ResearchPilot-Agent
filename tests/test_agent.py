"""Deterministic graph tests using fake LLM and tool dependencies."""

import json
from unittest import TestCase

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent import create_agent_graph, is_fast_query


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def planner_response():
    return AIMessage(content="1. Answer the question")


@tool
def fake_research(query: str) -> str:
    """Return deterministic research evidence."""
    return json.dumps(
        {
            "ok": True,
            "tool_name": "fake_research",
            "data": {"answer": "evidence"},
            "evidence": [
                {
                    "evidence_id": "WEB-1",
                    "source_type": "web",
                    "source_id": "example.test",
                    "excerpt": "evidence",
                    "score": None,
                }
            ],
        }
    )


class TestAgentGraph(TestCase):
    def test_fast_query_uses_one_direct_model_call(self):
        llm = FakeLLM([AIMessage(content="Paris.")])
        state = create_agent_graph(
            llm=llm, tools=[fake_research], rate_limit_delay=0
        ).invoke(
            {
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "fast_path": True,
            }
        )
        self.assertEqual(state["run_status"], "fast_completed")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(state["critic_status"], "skipped")

    def test_research_markers_disable_fast_path(self):
        self.assertFalse(is_fast_query("Research the latest result"))
        self.assertTrue(is_fast_query("What is 2 plus 2?"))

    def test_direct_answer_is_critic_approved(self):
        llm = FakeLLM(
            [
                planner_response(),
                AIMessage(content="Paris is the capital of France."),
                AIMessage(content="APPROVED"),
            ]
        )
        state = create_agent_graph(llm=llm, tools=[fake_research], rate_limit_delay=0).invoke(
            {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
        )
        self.assertEqual(state["run_status"], "completed")
        self.assertEqual(state["critic_status"], "approved")
        self.assertEqual(state["citation_status"], "not_required")

    def test_tool_call_round_trips_then_requires_evidence_citation(self):
        tool_call = {
            "name": "fake_research",
            "args": {"query": "latest result"},
            "id": "call-1",
            "type": "tool_call",
        }
        llm = FakeLLM(
            [
                planner_response(),
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content="The evidence supports this conclusion [WEB-1]."),
                AIMessage(content="APPROVED"),
            ]
        )
        state = create_agent_graph(llm=llm, tools=[fake_research], rate_limit_delay=0).invoke(
            {"messages": [{"role": "user", "content": "Research the latest result."}]}
        )
        self.assertEqual(state["run_status"], "completed")
        self.assertEqual(state["citation_status"], "present")
        self.assertTrue(any(message.type == "tool" for message in state["messages"]))

    def test_critic_failure_is_not_reported_as_approval(self):
        llm = FakeLLM(
            [
                planner_response(),
                AIMessage(content="A draft answer."),
                RuntimeError("critic service unavailable"),
            ]
        )
        state = create_agent_graph(llm=llm, tools=[fake_research], rate_limit_delay=0).invoke(
            {"messages": [{"role": "user", "content": "Give me an answer."}]}
        )
        self.assertEqual(state["run_status"], "critic_unavailable")
        self.assertEqual(state["critic_status"], "unavailable")

    def test_tool_limit_forces_synthesis(self):
        tool_call = {
            "name": "fake_research",
            "args": {"query": "repeat"},
            "id": "call-1",
            "type": "tool_call",
        }
        responses = [planner_response()]
        responses.extend(
            [
                AIMessage(content="", tool_calls=[tool_call]),
                AIMessage(content="Final answer [WEB-1]."),
            ]
        )
        responses.extend([AIMessage(content="APPROVED")])
        llm = FakeLLM(responses)
        state = create_agent_graph(
            llm=llm,
            tools=[fake_research],
            rate_limit_delay=0,
            max_tool_rounds=1,
        ).invoke({"messages": [{"role": "user", "content": "Use the source."}]})
        self.assertEqual(state["run_status"], "completed")
        self.assertGreaterEqual(len(llm.calls), 4)
