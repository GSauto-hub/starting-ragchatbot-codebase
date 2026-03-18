"""
Unit tests for AIGenerator, mocking the anthropic.Anthropic client.

These tests establish a baseline for the Claude API integration and catch regressions
in the tool execution loop.
"""
from unittest.mock import MagicMock, patch
import pytest

from ai_generator import AIGenerator


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


class TestAIGeneratorDirectResponse:

    def test_direct_response_no_tool_use(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client

            text_block = make_text_block("Python is a high-level programming language.")
            mock_client.messages.create.return_value = make_api_response(
                "end_turn", [text_block]
            )

            gen = AIGenerator(api_key="test-key", model="test-model")
            result = gen.generate_response(query="What is Python?")

            assert result == "Python is a high-level programming language."
            assert mock_client.messages.create.call_count == 1

    def test_conversation_history_in_system_prompt(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client

            mock_client.messages.create.return_value = make_api_response(
                "end_turn", [make_text_block("Follow-up answer.")]
            )

            gen = AIGenerator(api_key="test-key", model="test-model")
            history = "User: Hello\nAssistant: Hi there"
            gen.generate_response(query="Follow up question", conversation_history=history)

            call_kwargs = mock_client.messages.create.call_args[1]
            assert history in call_kwargs["system"]


class TestAIGeneratorToolExecution:

    def test_handle_tool_execution_calls_tool_manager(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client

            tool_block = make_tool_use_block("search_course_content", {"query": "MCP"})
            tool_response = make_api_response("tool_use", [tool_block])
            final_response = make_api_response(
                "end_turn", [make_text_block("MCP is a protocol.")]
            )
            mock_client.messages.create.side_effect = [tool_response, final_response]

            tool_manager = MagicMock()
            tool_manager.execute_tool.return_value = "MCP content result"

            gen = AIGenerator(api_key="test-key", model="test-model")
            result = gen.generate_response(
                query="What is MCP?",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="MCP"
            )
            assert result == "MCP is a protocol."

    def test_final_api_call_omits_tools(self):
        """
        The last-round API call must NOT include a 'tools' key.
        With MAX_TOOL_ROUNDS=2, round 0 is intermediate (includes tools),
        round 1 is the last round (omits tools).
        """
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client

            tool_block_1 = make_tool_use_block("search_course_content", {"query": "MCP"}, "toolu_001")
            tool_block_2 = make_tool_use_block("search_course_content", {"query": "MCP details"}, "toolu_002")
            tool_response_1 = make_api_response("tool_use", [tool_block_1])
            tool_response_2 = make_api_response("tool_use", [tool_block_2])
            final_response = make_api_response(
                "end_turn", [make_text_block("Final answer.")]
            )
            mock_client.messages.create.side_effect = [tool_response_1, tool_response_2, final_response]

            tool_manager = MagicMock()
            tool_manager.execute_tool.return_value = "content"

            gen = AIGenerator(api_key="test-key", model="test-model")
            gen.generate_response(
                query="test",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            assert mock_client.messages.create.call_count == 3
            third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
            assert "tools" not in third_call_kwargs


class TestAIGeneratorSequentialToolCalling:

    def _two_round_setup(self, mock_client, tool_manager):
        """Helper: configure mock for two tool rounds then a final text response."""
        tool_block_1 = make_tool_use_block("search_course_content", {"query": "topic"}, "toolu_001")
        tool_block_2 = make_tool_use_block("search_course_content", {"query": "related"}, "toolu_002")
        tool_response_1 = make_api_response("tool_use", [tool_block_1])
        tool_response_2 = make_api_response("tool_use", [tool_block_2])
        final_response = make_api_response("end_turn", [make_text_block("Final synthesized answer.")])
        mock_client.messages.create.side_effect = [tool_response_1, tool_response_2, final_response]
        tool_manager.execute_tool.return_value = "some content"

    def test_two_tool_rounds_makes_three_api_calls(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            tool_manager = MagicMock()
            self._two_round_setup(mock_client, tool_manager)

            gen = AIGenerator(api_key="test-key", model="test-model")
            result = gen.generate_response(
                query="Find a course covering the same topic as lesson X of course Y",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            assert mock_client.messages.create.call_count == 3
            assert tool_manager.execute_tool.call_count == 2
            assert result == "Final synthesized answer."

    def test_intermediate_api_call_includes_tools(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            tool_manager = MagicMock()
            self._two_round_setup(mock_client, tool_manager)

            gen = AIGenerator(api_key="test-key", model="test-model")
            gen.generate_response(
                query="multi-step query",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            assert "tools" in second_call_kwargs

    def test_final_api_call_of_two_rounds_omits_tools(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            tool_manager = MagicMock()
            self._two_round_setup(mock_client, tool_manager)

            gen = AIGenerator(api_key="test-key", model="test-model")
            gen.generate_response(
                query="multi-step query",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
            assert "tools" not in third_call_kwargs

    def test_early_exit_when_no_tool_use_after_round_one(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client

            tool_block = make_tool_use_block("search_course_content", {"query": "MCP"})
            tool_response = make_api_response("tool_use", [tool_block])
            end_turn_response = make_api_response("end_turn", [make_text_block("Done after one round.")])
            mock_client.messages.create.side_effect = [tool_response, end_turn_response]

            tool_manager = MagicMock()
            tool_manager.execute_tool.return_value = "content"

            gen = AIGenerator(api_key="test-key", model="test-model")
            result = gen.generate_response(
                query="simple query",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            assert mock_client.messages.create.call_count == 2
            assert result == "Done after one round."

    def test_tool_error_returns_response_without_raising(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client

            tool_block = make_tool_use_block("search_course_content", {"query": "fail"})
            tool_response = make_api_response("tool_use", [tool_block])
            final_response = make_api_response("end_turn", [make_text_block("Recovered response.")])
            mock_client.messages.create.side_effect = [tool_response, final_response]

            tool_manager = MagicMock()
            tool_manager.execute_tool.side_effect = ValueError("DB down")

            gen = AIGenerator(api_key="test-key", model="test-model")
            result = gen.generate_response(
                query="query that errors",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            assert isinstance(result, str)
            assert result == "Recovered response."

    def test_full_message_history_sent_in_final_call(self):
        with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            tool_manager = MagicMock()
            self._two_round_setup(mock_client, tool_manager)

            gen = AIGenerator(api_key="test-key", model="test-model")
            gen.generate_response(
                query="multi-step query",
                tools=[{"name": "search_course_content"}],
                tool_manager=tool_manager,
            )

            third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
            messages = third_call_kwargs["messages"]
            # user query, assistant turn 1, tool results 1, assistant turn 2, tool results 2
            assert len(messages) == 5
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[3]["role"] == "assistant"
            assert messages[4]["role"] == "user"
