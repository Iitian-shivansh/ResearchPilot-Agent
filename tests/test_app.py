"""Streamlit smoke tests: drive app.py with a fake agent graph (no API keys, no network)."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import src.agent

APP = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeGraph:
    def __init__(self, events):
        self.events = events
        self.inputs = []

    def stream(self, state):
        self.inputs.append(state)
        yield from self.events


def _submit(graph, text="Research the latest result"):
    st.cache_resource.clear()
    with patch.object(src.agent, "create_agent_graph", return_value=graph):
        app = AppTest.from_file(APP, default_timeout=30)
        app.run()
        app.chat_input[0].set_value(text).run()
    st.cache_resource.clear()
    return app


class TestStreamlitApp(TestCase):
    def test_verified_answer_is_shown_and_kept_in_conversation(self):
        graph = FakeGraph([
            {"planner": {"run_status": "planned"}},
            {"executor": {"run_status": "executing", "messages": [AIMessage(content="Answer [WEB-1]")]}},
            {"critic": {"run_status": "completed", "critic_status": "approved"}},
        ])
        app = _submit(graph)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("Answer [WEB-1]", [m.value for m in app.markdown])
        self.assertEqual(app.session_state.conversation[-1]["content"], "Answer [WEB-1]")
        self.assertEqual(graph.inputs[0]["mode"], "Deep")

    def test_failure_reason_survives_rerun(self):
        graph = FakeGraph([
            {"planner": {"run_status": "planned"}},
            {"executor": {
                "run_status": "executor_failed",
                "messages": [AIMessage(content="Executor failed: Authentication failed. Check GROQ_API_KEY.")],
            }},
        ])
        app = _submit(graph)
        self.assertFalse(app.exception)
        errors = [e.value for e in app.error]
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("executor_failed", errors[0])
        self.assertIn("GROQ_API_KEY", errors[0])
        self.assertEqual(app.session_state.conversation[-1]["role"], "user")

    def test_unapproved_draft_is_not_shown(self):
        graph = FakeGraph([
            {"planner": {"run_status": "planned"}},
            {"executor": {"run_status": "executing", "messages": [AIMessage(content="Unverified draft")]}},
            {"critic": {"run_status": "critic_rejected", "critic_status": "rejected"}},
        ])
        app = _submit(graph)
        self.assertNotIn("Unverified draft", [m.value for m in app.markdown])
        self.assertIn("rejected", app.error[0].value)
