"""
Tests for FastAPI endpoints in app.py.

app.py has two module-level side effects that break a plain import in tests:
  1. RAGSystem() is instantiated, requiring ChromaDB and a real API key.
  2. app.mount() is called with DevStaticFiles pointing to ../frontend, which
     does not exist in the test environment.

Strategy: before importing app we patch
  - `rag_system.RAGSystem`  → returns a MagicMock
  - `fastapi.staticfiles.StaticFiles` → replaced with a no-op class so
    DevStaticFiles (which inherits from it) can be instantiated without a real
    directory.  API routes are matched before the static-files mount, so the
    no-op implementation is never invoked for /api/* requests.

The module-scoped `_app_module` fixture imports app exactly once; each test
gets a fresh `mock_rag_system` (function-scoped) injected into
`app.rag_system` so tests cannot interfere with each other.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient


class _NoOpStaticFiles:
    """Replaces StaticFiles so DevStaticFiles can be instantiated in tests."""

    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, scope, receive, send):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _app_module():
    """Import app.py once per module with side effects neutralised."""
    sys.modules.pop("app", None)

    _dummy_rag = MagicMock()

    with patch("fastapi.staticfiles.StaticFiles", _NoOpStaticFiles), \
         patch("rag_system.RAGSystem", return_value=_dummy_rag):
        import app as _app  # noqa: PLC0415

    yield _app

    sys.modules.pop("app", None)


@pytest.fixture
def mock_rag(_app_module):
    """Fresh RAGSystem mock injected into the imported app module."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "generated-session"
    mock.query.return_value = (
        "Default answer.",
        [{"text": "Source text", "url": "https://example.com/lesson/1"}],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    _app_module.rag_system = mock
    return mock


@pytest.fixture
def client(_app_module, mock_rag):
    """Starlette TestClient wrapping the real FastAPI app."""
    return TestClient(_app_module.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /api/query  (POST)
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_returns_answer_and_session_id(self, client, mock_rag):
        mock_rag.query.return_value = (
            "MCP is a protocol for tool calling.",
            [{"text": "Intro to MCP – Lesson 1", "url": "https://example.com/mcp/1"}],
        )
        mock_rag.session_manager.create_session.return_value = "sess-001"

        response = client.post("/api/query", json={"query": "What is MCP?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "MCP is a protocol for tool calling."
        assert data["session_id"] == "sess-001"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Intro to MCP – Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/mcp/1"

    def test_creates_session_when_none_provided(self, client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "new-session"
        mock_rag.query.return_value = ("Answer.", [])

        response = client.post("/api/query", json={"query": "hello"})

        assert response.status_code == 200
        assert response.json()["session_id"] == "new-session"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer.", [])

        response = client.post(
            "/api/query", json={"query": "hello", "session_id": "existing-sess"}
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "existing-sess"
        # Session should be reused, not created
        mock_rag.session_manager.create_session.assert_not_called()

    def test_empty_sources_list_is_valid(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer with no sources.", [])

        response = client.post("/api/query", json={"query": "simple question"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_missing_query_field_returns_422(self, client, mock_rag):
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_empty_query_string_is_accepted(self, client, mock_rag):
        mock_rag.query.return_value = ("Empty query response.", [])

        response = client.post("/api/query", json={"query": ""})

        assert response.status_code == 200

    def test_rag_exception_returns_500(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("vector store unavailable")

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 500
        assert "vector store unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /api/courses  (GET)
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_returns_course_stats(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["MCP Basics", "Advanced RAG", "Prompt Engineering"],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert data["course_titles"] == [
            "MCP Basics",
            "Advanced RAG",
            "Prompt Engineering",
        ]

    def test_returns_empty_catalog(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_exception_returns_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("db offline")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "db offline" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /api/session/{session_id}  (DELETE)
# ---------------------------------------------------------------------------

class TestSessionEndpoint:

    def test_delete_session_returns_ok(self, client, mock_rag):
        response = client.delete("/api/session/my-session-id")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_delete_session_calls_clear(self, client, mock_rag):
        client.delete("/api/session/target-session")

        mock_rag.session_manager.clear_session.assert_called_once_with(
            "target-session"
        )
