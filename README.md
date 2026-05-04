# 🧠 LangChain Learning Project — Smart Research Assistant

Learn LangChain by building a real project, one concept at a time.

## Setup
```bash
pip install -r requirements.txt
```

Add your API key to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

## Modules (Run in Order)

| # | Module | Concept | Run Command |
|---|--------|---------|-------------|
| 1 | Chains & LCEL | Prompt → Model → Parser pipeline | `python module_1_chains/basic_chain.py` |
| 2 | Structured Output | Pydantic models + output parsers | `python module_2_structured_output/structured_output.py` |
| 3 | Memory | Conversational context & sessions | `python module_3_memory/chat_with_memory.py` |
| 4 | Tools & Agents | @tool, ReAct agent, web search | `python module_4_agents/research_agent.py` |
| 5a | RAG Ingestion | Load, chunk, embed, store documents | `python module_5_rag/ingest.py` |
| 5b | RAG Query | Document Q&A with retrieval chain | `python module_5_rag/query.py` |

## Key Concepts Map

```
LangChain Ecosystem
├── Chains (LCEL)          ← Module 1: prompt | model | parser
├── Output Parsers         ← Module 2: raw text → structured data
├── Memory                 ← Module 3: stateless → stateful
├── Tools & Agents         ← Module 4: dynamic tool selection
└── RAG                    ← Module 5: your documents + LLM
    ├── Document Loaders
    ├── Text Splitters
    ├── Embeddings
    └── Vector Stores
```

## Tech Stack
- **LLM**: Google Gemini 2.0 Flash
- **Framework**: LangChain + LangGraph
- **Vector DB**: FAISS (local)
- **Search**: DuckDuckGo (free, no key needed)
