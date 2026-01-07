# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv run` to execute Python files. Do not use pip directly.**

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh

# Run manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app runs at http://localhost:8000 with API docs at http://localhost:8000/docs.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for educational course content. It uses Claude AI with tool calling to search course materials stored in ChromaDB.

### Backend Components (backend/)

**Request Flow:** `app.py` → `rag_system.py` → `ai_generator.py` → `search_tools.py` → `vector_store.py`

- **app.py**: FastAPI server with two endpoints: `POST /api/query` (process questions) and `GET /api/courses` (get course stats). Serves frontend static files.

- **rag_system.py**: Main orchestrator. Coordinates document ingestion, query processing, session management, and tool execution. Entry point for all RAG operations.

- **ai_generator.py**: Claude API integration. Sends queries with tool definitions, handles tool execution loop (Claude requests tool → execute → return results → get final response).

- **search_tools.py**: Implements Anthropic tool calling pattern. `CourseSearchTool` wraps vector store searches. `ToolManager` registers tools and tracks sources for UI attribution.

- **vector_store.py**: ChromaDB wrapper with two collections:
  - `course_catalog`: Course metadata for fuzzy name resolution
  - `course_content`: Chunked course text with embeddings (all-MiniLM-L6-v2)

- **document_processor.py**: Parses course documents, extracts metadata (title, instructor, lessons), chunks text with sentence-aware splitting.

- **session_manager.py**: Tracks conversation history per session for multi-turn context.

- **models.py**: Pydantic models for `Course`, `Lesson`, `CourseChunk`.

- **config.py**: Centralized settings (API keys, model names, chunk sizes).

### Frontend (frontend/)

Vanilla HTML/CSS/JS chat interface. Uses marked.js for markdown rendering. Communicates via fetch to `/api/query`.

### Data Flow

1. On startup, `docs/*.txt` files are parsed, chunked, embedded, and stored in ChromaDB
2. User query → FastAPI → RAGSystem → Claude API with `search_course_content` tool
3. Claude decides whether to search → tool executes vector search → results returned to Claude
4. Claude synthesizes answer → response with sources sent to frontend

### Course Document Format

Documents in `docs/` must follow this structure for proper parsing:
```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[content...]
```
