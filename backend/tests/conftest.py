import sys
import os
import tempfile
import pytest
from unittest.mock import MagicMock

# Add backend directory to Python path so tests can import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from config import Config


@pytest.fixture
def sample_course_chunk():
    return CourseChunk(
        content="MCP stands for Model Context Protocol. It enables AI models to interact with tools and external services.",
        course_title="Introduction to MCP",
        lesson_number=1,
        chunk_index=0,
    )


@pytest.fixture
def sample_course():
    return Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp",
        instructor="Jane Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/mcp/1",
            ),
            Lesson(
                lesson_number=2,
                title="Tool Calling",
                lesson_link="https://example.com/mcp/2",
            ),
        ],
    )


@pytest.fixture
def mock_search_results():
    return SearchResults(
        documents=["MCP stands for Model Context Protocol. It enables AI models to interact with tools."],
        metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
        distances=[0.1],
    )


@pytest.fixture
def mock_rag_system():
    """A fully-mocked RAGSystem usable across all test files."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-id"
    mock.query.return_value = (
        "Test answer.",
        [{"text": "Source text", "url": "https://example.com/1"}],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return mock


@pytest.fixture
def test_config(tmp_path):
    """A Config instance suitable for unit/integration tests."""
    return Config(
        MAX_RESULTS=5,
        CHROMA_PATH=str(tmp_path / "chroma"),
        ANTHROPIC_API_KEY="test-key",
    )


@pytest.fixture
def in_memory_vector_store(sample_course, sample_course_chunk):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(
            chroma_path=tmpdir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        store.add_course_metadata(sample_course)
        store.add_course_content([sample_course_chunk])
        yield store
