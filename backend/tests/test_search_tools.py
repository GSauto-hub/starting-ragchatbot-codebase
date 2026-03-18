"""
Unit tests for CourseSearchTool and CourseOutlineTool using a mock VectorStore.

Expected failure: test_execute_search_error shows that when VectorStore.search()
returns an error (triggered by MAX_RESULTS=0), the raw error string is passed
directly to the AI — meaning every real query hits this broken path.
"""
from unittest.mock import MagicMock
import pytest

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


def make_mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course"
    return store


class TestCourseSearchTool:

    def test_execute_returns_formatted_results(self, mock_search_results):
        store = make_mock_store()
        store.search.return_value = mock_search_results

        tool = CourseSearchTool(store)
        result = tool.execute(query="MCP basics")

        assert "Introduction to MCP" in result
        assert "Lesson 1" in result
        assert "Model Context Protocol" in result

    def test_execute_empty_results(self):
        store = make_mock_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

        tool = CourseSearchTool(store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_search_error(self):
        """
        Exposes the bug: when MAX_RESULTS=0, ChromaDB raises an error which
        VectorStore.search() catches and returns as SearchResults.empty(error_msg).
        CourseSearchTool.execute() then returns that raw error string to the AI.
        """
        store = make_mock_store()
        error_msg = "Search error: n_results must be a positive integer, got: 0"
        store.search.return_value = SearchResults.empty(error_msg)

        tool = CourseSearchTool(store)
        result = tool.execute(query="what is MCP?")

        # The raw error string is returned to Claude instead of real content
        assert result == error_msg
        assert "Search error" in result

    def test_last_sources_populated(self, mock_search_results):
        store = make_mock_store()
        store.search.return_value = mock_search_results

        tool = CourseSearchTool(store)
        tool.execute(query="MCP basics")

        assert len(tool.last_sources) > 0
        assert "text" in tool.last_sources[0]
        assert "url" in tool.last_sources[0]

    def test_last_sources_empty_on_no_results(self):
        store = make_mock_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

        # Fresh tool instance starts with empty last_sources
        tool = CourseSearchTool(store)
        tool.execute(query="nonexistent topic")

        # Empty results do not populate last_sources
        assert tool.last_sources == []

    def test_course_filter(self):
        store = make_mock_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

        tool = CourseSearchTool(store)
        tool.execute(query="MCP", course_name="Introduction to MCP")

        store.search.assert_called_once_with(
            query="MCP",
            course_name="Introduction to MCP",
            lesson_number=None,
        )


class TestCourseOutlineTool:

    def test_outline_tool_returns_structured_text(self):
        store = MagicMock()
        store.get_course_outline.return_value = {
            "title": "Introduction to MCP",
            "course_link": "https://example.com/mcp",
            "lessons": [
                {"lesson_number": 1, "lesson_title": "Getting Started"},
                {"lesson_number": 2, "lesson_title": "Tool Calling"},
            ],
        }

        tool = CourseOutlineTool(store)
        result = tool.execute(course_title="MCP")

        assert "Introduction to MCP" in result
        assert "Lessons:" in result
        assert "Lesson 1" in result
        assert "Getting Started" in result
        assert "Lesson 2" in result
        assert "Tool Calling" in result

    def test_outline_tool_not_found(self):
        store = MagicMock()
        store.get_course_outline.return_value = None

        tool = CourseOutlineTool(store)
        result = tool.execute(course_title="Nonexistent Course")

        assert "No course found" in result
        assert "Nonexistent Course" in result


class TestToolManager:

    def test_register_and_execute_tool(self, mock_search_results):
        store = make_mock_store()
        store.search.return_value = mock_search_results

        manager = ToolManager()
        manager.register_tool(CourseSearchTool(store))

        result = manager.execute_tool("search_course_content", query="MCP")
        assert "Introduction to MCP" in result

    def test_execute_unknown_tool(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")
        assert "not found" in result

    def test_get_tool_definitions(self):
        store = make_mock_store()
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(store))

        defs = manager.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"

    def test_reset_sources(self, mock_search_results):
        store = make_mock_store()
        store.search.return_value = mock_search_results

        manager = ToolManager()
        search_tool = CourseSearchTool(store)
        manager.register_tool(search_tool)

        manager.execute_tool("search_course_content", query="MCP")
        assert len(manager.get_last_sources()) > 0

        manager.reset_sources()
        assert manager.get_last_sources() == []
