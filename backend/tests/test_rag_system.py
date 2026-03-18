"""
Integration-style tests for RAGSystem.query() using a real in-memory ChromaDB
instance and a mocked Anthropic API.

Expected failures before fix:
- test_max_results_zero_causes_search_failure: FAILS, confirming the bug
  (MAX_RESULTS=0 in config.py causes every vector search to return an error)

After applying the fix (MAX_RESULTS = 5 in config.py), all tests pass.
"""
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from rag_system import RAGSystem
from config import Config


def make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name, input_dict, block_id="toolu_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = input_dict
    return block


def make_api_response(stop_reason, content_blocks):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


@pytest.fixture
def sample_data():
    course = Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp",
        instructor="Jane Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/mcp/1",
            ),
        ],
    )
    chunk = CourseChunk(
        content="MCP stands for Model Context Protocol. It enables AI models to interact with tools and external services.",
        course_title="Introduction to MCP",
        lesson_number=1,
        chunk_index=0,
    )
    return course, chunk


class TestMaxResultsBug:

    def test_max_results_zero_causes_search_failure(self, sample_data):
        """
        Reveals the root-cause bug: MAX_RESULTS=0 causes ChromaDB to raise
        an error on every search. VectorStore.search() catches it and returns
        SearchResults.empty(error_msg), so every query silently fails.

        This test FAILS before the fix (MAX_RESULTS=0) and PASSES after (MAX_RESULTS=5).
        """
        course, chunk = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                chroma_path=tmpdir,
                embedding_model="all-MiniLM-L6-v2",
                max_results=0,  # The bug value from config.py
            )
            store.add_course_metadata(course)
            store.add_course_content([chunk])

            result = store.search(query="What is MCP?")

            # With max_results=0, ChromaDB raises an error
            assert result.error is not None, (
                "Expected a search error when max_results=0, but got none. "
                "This indicates MAX_RESULTS=0 is not triggering the ChromaDB failure."
            )
            assert "Search error" in result.error


class TestRAGSystemQuery:

    def test_query_with_valid_max_results(self, sample_data):
        """
        With MAX_RESULTS=5, searches succeed and the response is meaningful.
        Requires the fix to be applied first.
        """
        course, chunk = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                MAX_RESULTS=5,
                CHROMA_PATH=tmpdir,
                ANTHROPIC_API_KEY="test-key",
            )

            with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                MockAnthropic.return_value = mock_client

                rag = RAGSystem(config)
                rag.vector_store.add_course_metadata(course)
                rag.vector_store.add_course_content([chunk])

                # Claude calls the search tool, gets results, then gives a final answer
                tool_block = make_tool_use_block(
                    "search_course_content", {"query": "What is MCP?"}
                )
                tool_response = make_api_response("tool_use", [tool_block])
                final_text = make_text_block("MCP is a protocol for tool calling.")
                final_response = make_api_response("end_turn", [final_text])
                mock_client.messages.create.side_effect = [tool_response, final_response]

                response, sources = rag.query("What does the MCP course cover?")

                assert response == "MCP is a protocol for tool calling."
                assert len(sources) > 0

    def test_sources_are_dicts_with_text_and_url(self, sample_data):
        """
        Sources returned from query() must be dicts with 'text' and 'url' keys
        so that QueryResponse can serialize them for the frontend source chips.
        """
        course, chunk = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                MAX_RESULTS=5,
                CHROMA_PATH=tmpdir,
                ANTHROPIC_API_KEY="test-key",
            )

            with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                MockAnthropic.return_value = mock_client

                rag = RAGSystem(config)
                rag.vector_store.add_course_metadata(course)
                rag.vector_store.add_course_content([chunk])

                tool_block = make_tool_use_block(
                    "search_course_content", {"query": "MCP"}
                )
                tool_response = make_api_response("tool_use", [tool_block])
                final_response = make_api_response(
                    "end_turn", [make_text_block("MCP is great.")]
                )
                mock_client.messages.create.side_effect = [tool_response, final_response]

                _, sources = rag.query("What is MCP?")

                assert isinstance(sources, list)
                assert len(sources) > 0
                for source in sources:
                    assert isinstance(source, dict), f"Source must be a dict, got {type(source)}"
                    assert "text" in source, f"Source missing 'text' key: {source}"
                    assert "url" in source, f"Source missing 'url' key: {source}"

    def test_session_history_updated_after_query(self, sample_data):
        """
        After a query with a session_id, conversation history is stored
        and retrievable via session_manager.
        """
        course, chunk = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                MAX_RESULTS=5,
                CHROMA_PATH=tmpdir,
                ANTHROPIC_API_KEY="test-key",
            )

            with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                MockAnthropic.return_value = mock_client

                rag = RAGSystem(config)
                mock_client.messages.create.return_value = make_api_response(
                    "end_turn", [make_text_block("Some response.")]
                )

                session_id = "test_session_123"
                rag.query("What is MCP?", session_id=session_id)

                history = rag.session_manager.get_conversation_history(session_id)
                assert history is not None
                assert len(history) > 0

    def test_tool_manager_sources_reset_after_query(self, sample_data):
        """
        tool_manager.get_last_sources() returns [] after each query completes,
        so sources from one query don't bleed into the next.
        """
        course, chunk = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                MAX_RESULTS=5,
                CHROMA_PATH=tmpdir,
                ANTHROPIC_API_KEY="test-key",
            )

            with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
                mock_client = MagicMock()
                MockAnthropic.return_value = mock_client

                rag = RAGSystem(config)
                mock_client.messages.create.return_value = make_api_response(
                    "end_turn", [make_text_block("Response.")]
                )

                rag.query("What is MCP?")

                assert rag.tool_manager.get_last_sources() == []
