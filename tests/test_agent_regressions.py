"""Regression tests for agent defects found in the repository audit."""

import json
from unittest import TestCase

import groq
import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent import _is_rate_limit, create_agent_graph, describe_api_error

EXECUTED: list[str] = []


@tool
def colliding_research(query: str) -> str:
    """Return evidence that always uses WEB-1, like per-call provider numbering."""
    EXECUTED.append(query)
    return json.dumps({
        "ok": True, "tool_name": "colliding_research", "data": {},
        "evidence": [{
            "evidence_id": "WEB-1", "source_type": "web",
            "source_id": f"https://example.test/{query}", "excerpt": "text",
        }],
    })


class AlwaysResearchLLM:
    """Requests a new tool call on every executor turn until the budget forces synthesis."""

    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        system = str(messages[0].content)
        if "planning assistant" in system:
            return AIMessage(content="1. Research")
        if "strict research reviewer" in system:
            return AIMessage(content='{"decision":"approved","feedback":""}')
        if "forced synthesis" in system:
            return AIMessage(content="Synthesized answer [WEB-1].")
        n = len(self.calls)
        return AIMessage(content="", tool_calls=[{
            "name": "colliding_research", "args": {"query": f"q{n}"}, "id": f"call-{n}",
        }])


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls("provider error", response=httpx.Response(status, request=request), body=None)


class TestToolMessageDeduplication(TestCase):
    def setUp(self):
        EXECUTED.clear()

    def test_renumbered_evidence_does_not_duplicate_tool_messages(self):
        state = create_agent_graph(
            llm=AlwaysResearchLLM(), tools=[colliding_research], rate_limit_delay=0,
        ).invoke({"messages": [{"role": "user", "content": "Research x"}], "mode": "Deep"})

        tool_messages = [m for m in state["messages"] if m.type == "tool"]
        call_ids = [m.tool_call_id for m in tool_messages]
        self.assertEqual(len(call_ids), len(set(call_ids)), f"duplicate tool messages: {call_ids}")
        self.assertEqual(len(tool_messages), len(EXECUTED))
        evidence_ids = [json.loads(m.content)["evidence"][0]["evidence_id"] for m in tool_messages]
        self.assertEqual(len(evidence_ids), len(set(evidence_ids)), evidence_ids)

    def test_mode_tool_budget_counts_real_executions(self):
        for mode, budget in (("Quick", 2), ("Analyze", 3), ("Deep", 4)):
            with self.subTest(mode=mode):
                EXECUTED.clear()
                create_agent_graph(
                    llm=AlwaysResearchLLM(), tools=[colliding_research], rate_limit_delay=0,
                ).invoke({"messages": [{"role": "user", "content": "Research x"}], "mode": mode})
                self.assertEqual(len(EXECUTED), budget)

    def test_llm_never_receives_duplicate_tool_results(self):
        llm = AlwaysResearchLLM()
        create_agent_graph(
            llm=llm, tools=[colliding_research], rate_limit_delay=0,
        ).invoke({"messages": [{"role": "user", "content": "Research x"}], "mode": "Deep"})
        for sent in llm.calls:
            ids = [m.tool_call_id for m in sent if getattr(m, "type", "") == "tool"]
            self.assertEqual(len(ids), len(set(ids)), ids)


class TestProviderErrorHandling(TestCase):
    def test_rate_limit_detection_is_specific(self):
        self.assertTrue(_is_rate_limit(_status_error(groq.RateLimitError, 429)))
        self.assertTrue(_is_rate_limit(RuntimeError("429 Too Many Requests")))
        self.assertTrue(_is_rate_limit(RuntimeError("Rate limit reached for model")))
        self.assertFalse(_is_rate_limit(RuntimeError("Failed to generate response")))
        self.assertFalse(_is_rate_limit(RuntimeError("separate validation error")))

    def test_error_descriptions_are_actionable_and_hide_raw_messages(self):
        auth = describe_api_error(_status_error(groq.AuthenticationError, 401))
        self.assertIn("GROQ_API_KEY", auth)
        self.assertIn("Rate limit", describe_api_error(_status_error(groq.RateLimitError, 429)))
        secretish = RuntimeError("upstream said: key=abc123 invalid")
        self.assertNotIn("abc123", describe_api_error(secretish))

    def test_executor_failure_message_explains_cause(self):
        class FailingExecutorLLM(AlwaysResearchLLM):
            def invoke(self, messages):
                if "research executor" in str(messages[0].content):
                    raise _status_error(groq.AuthenticationError, 401)
                return super().invoke(messages)

        state = create_agent_graph(
            llm=FailingExecutorLLM(), tools=[colliding_research], rate_limit_delay=0,
        ).invoke({"messages": [{"role": "user", "content": "Research x"}]})
        self.assertEqual(state["run_status"], "executor_failed")
        self.assertIn("GROQ_API_KEY", state["messages"][-1].content)

    def test_critic_rate_limit_retry_keeps_reviewer_instructions(self):
        class RateLimitedCriticLLM(AlwaysResearchLLM):
            critic_attempts = 0

            def invoke(self, messages):
                if "strict research reviewer" in str(messages[0].content):
                    self.critic_attempts += 1
                    if self.critic_attempts == 1:
                        raise _status_error(groq.RateLimitError, 429)
                if "research executor" in str(messages[0].content):
                    self.calls.append(messages)
                    return AIMessage(content="Direct answer with no tools.")
                return super().invoke(messages)

        llm = RateLimitedCriticLLM()
        state = create_agent_graph(llm=llm, tools=[colliding_research], rate_limit_delay=0).invoke(
            {"messages": [{"role": "user", "content": "Explain x"}]}
        )
        self.assertEqual(llm.critic_attempts, 2)
        self.assertEqual(state["critic_status"], "approved")
