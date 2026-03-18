import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to 2 sequential tool calls per query** — use a second tool call only when the first result is insufficient to fully answer the question
- For course outline queries (e.g. "what lessons does X have?", "show me the outline of X"):
  use the `get_course_outline` tool and return the course title, course link, and each lesson's number and title
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, api_params, tools, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _run_tool_loop(self, response, api_params: Dict[str, Any], tools, tool_manager):
        """
        Run up to MAX_TOOL_ROUNDS sequential tool call rounds.

        Args:
            response: The initial tool_use response from Claude
            api_params: API parameters from the initial call
            tools: Tool definitions to pass on intermediate rounds
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = api_params["messages"].copy()

        for round_idx in range(self.MAX_TOOL_ROUNDS):
            # Append Claude's tool-use assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool_use blocks, catching errors
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Append tool results as user turn
            messages.append({"role": "user", "content": tool_results})

            # Build next params — include tools only if more rounds remain
            is_last_round = (round_idx == self.MAX_TOOL_ROUNDS - 1)
            next_params = {**self.base_params, "messages": messages, "system": api_params["system"]}
            if not is_last_round:
                next_params["tools"] = tools
                next_params["tool_choice"] = {"type": "auto"}

            # Get next response
            response = self.client.messages.create(**next_params)

            # Early exit if no more tool calls
            if response.stop_reason != "tool_use":
                break

        return response.content[0].text