"""
╔══════════════════════════════════════════════════════════════╗
║  AI Research Assistant — FastAPI Backend                      ║
╠══════════════════════════════════════════════════════════════╣
║  Serves the frontend and provides API endpoints for all      ║
║  LangChain modes: Chat, Research, RAG, Analyze.              ║
║                                                              ║
║  Run:  python app.py                                         ║
║  Open: http://localhost:8000                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, json, math, asyncio
sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pathlib import Path

from config import get_llm

# ═══════════════════════════════════════════════════════════════
#  LangChain imports
# ═══════════════════════════════════════════════════════════════
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="AI Research Assistant")

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Memory Store (shared across all modes)
# ═══════════════════════════════════════════════════════════════
memory_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = InMemoryChatMessageHistory()
    return memory_store[session_id]

# ═══════════════════════════════════════════════════════════════
#  MODE 1: Chat (conversational with memory)
# ═══════════════════════════════════════════════════════════════
chat_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful, friendly AI assistant. "
     "Keep responses concise but thorough. Use markdown formatting. "
     "Remember details from earlier in the conversation."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

chat_chain = chat_prompt | get_llm() | StrOutputParser()

chat_with_memory = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ═══════════════════════════════════════════════════════════════
#  MODE 2: Research Agent (tools: search, calc, word count)
# ═══════════════════════════════════════════════════════════════
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression.
    Use this for math, arithmetic, or calculations.
    Input: valid math like '2 + 2' or 'sqrt(144)'.
    """
    try:
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "pi": math.pi,
            "e": math.e, "abs": abs, "pow": pow, "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def word_counter(text: str) -> str:
    """Count words, characters, and sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo.
    Use for real-time info, current events, or facts.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            output = []
            for r in results:
                output.append(f"**{r['title']}**\n{r['body']}\n")
            return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"

agent_tools = [calculator, web_search, word_counter]
research_agent = create_react_agent(
    get_llm(temperature=0),
    agent_tools,
    prompt="You are a helpful research assistant with access to tools. "
           "Use the appropriate tool for each task. "
           "For math, use calculator. For current info, use web_search. "
           "Always provide clear explanations with your results. Use markdown."
)

# ═══════════════════════════════════════════════════════════════
#  MODE 3: RAG (document Q&A)
# ═══════════════════════════════════════════════════════════════
rag_retriever = None

def get_rag_retriever():
    global rag_retriever
    if rag_retriever is not None:
        return rag_retriever
    index_path = Path(__file__).parent / "module_5_rag" / "faiss_index"
    if not index_path.exists():
        return None
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    rag_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return rag_retriever

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY from the context below. "
     "If the context doesn't contain the answer, say 'I don't have enough information in my documents to answer that.' "
     "Use markdown formatting."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# ═══════════════════════════════════════════════════════════════
#  MODE 4: Analyze (structured output)
# ═══════════════════════════════════════════════════════════════
class AnalysisResult(BaseModel):
    """Structured analysis of a topic."""
    title: str = Field(description="Concise title")
    summary: str = Field(description="2-3 sentence summary")
    key_points: List[str] = Field(description="3-5 key takeaways")
    difficulty_level: str = Field(description="beginner, intermediate, or advanced")
    related_topics: List[str] = Field(description="2-3 related topics")

analyze_parser = PydanticOutputParser(pydantic_object=AnalysisResult)

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research analyst. Analyze the topic and return structured data.\n\n"
     "{format_instructions}"),
    ("human", "Analyze: {topic}")
]).partial(format_instructions=analyze_parser.get_format_instructions())

analyze_chain = analyze_prompt | get_llm(temperature=0.3) | analyze_parser

# ═══════════════════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = static_dir / "index.html"
    return FileResponse(html_path)

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    message = body.get("message", "")
    mode = body.get("mode", "chat")
    session_id = body.get("session_id", "default")

    async def event_stream():
        try:
            if mode == "chat":
                config = {"configurable": {"session_id": session_id}}
                for chunk in chat_with_memory.stream(
                    {"input": message}, config=config
                ):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            elif mode == "research":
                result = research_agent.invoke({
                    "messages": [HumanMessage(content=message)]
                })
                messages = result["messages"]
                # Send tool usage events
                for msg in messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name'], 'args': tc['args']})}\n\n"
                    elif msg.type == "tool":
                        content = msg.content[:300] if len(msg.content) > 300 else msg.content
                        yield f"data: {json.dumps({'type': 'tool_result', 'content': content})}\n\n"
                # Send final response
                final = messages[-1].content
                yield f"data: {json.dumps({'type': 'token', 'content': final})}\n\n"

            elif mode == "rag":
                retriever = get_rag_retriever()
                if retriever is None:
                    yield f"data: {json.dumps({'type': 'token', 'content': '⚠️ RAG index not found. Run `python module_5_rag/ingest.py` first to build the document index.'})}\n\n"
                else:
                    # Retrieve docs
                    docs = retriever.invoke(message)
                    sources = [d.page_content[:100] + "..." for d in docs]
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
                    # Build and run RAG chain
                    context = format_docs(docs)
                    rag_chain = rag_prompt | get_llm(temperature=0.3) | StrOutputParser()
                    for chunk in rag_chain.stream({"context": context, "question": message}):
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            elif mode == "analyze":
                yield f"data: {json.dumps({'type': 'status', 'content': 'Analyzing...'})}\n\n"
                result = analyze_chain.invoke({"topic": message})
                analysis = {
                    "title": result.title,
                    "summary": result.summary,
                    "key_points": result.key_points,
                    "difficulty_level": result.difficulty_level,
                    "related_topics": result.related_topics,
                }
                yield f"data: {json.dumps({'type': 'analysis', 'content': analysis})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/sessions")
async def list_sessions():
    sessions = {}
    for sid, history in memory_store.items():
        sessions[sid] = {"message_count": len(history.messages)}
    return sessions

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    if session_id in memory_store:
        del memory_store[session_id]
        return {"status": "cleared"}
    return {"status": "not_found"}

# ═══════════════════════════════════════════════════════════════
#  Run Server
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  🚀 AI Research Assistant")
    print("  Open: http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
