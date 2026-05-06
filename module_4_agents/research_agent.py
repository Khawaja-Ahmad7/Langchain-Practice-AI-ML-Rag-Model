"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 4: Tools & Agents                                    ║
╠══════════════════════════════════════════════════════════════╣
║  A chain is fixed: A → B → C. An agent is DYNAMIC: it        ║
║  decides which tools to use based on the question.           ║
║                                                              ║
║  ReAct pattern: THINK → ACT → OBSERVE → RESPOND              ║
╚══════════════════════════════════════════════════════════════╝
"""

import math, sys
sys.path.insert(0, sys.path[0] + '/..')
sys.stdout.reconfigure(encoding='utf-8')

from config import get_llm
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# ── LLM (from shared config.py) ────────────────────────────────
model = get_llm(temperature=0)  # Agents need deterministic behavior

# ── Define Tools ────────────────────────────────────────────────
# @tool turns a function into a LangChain tool. The LLM reads
# the DOCSTRING to decide when to use each tool!

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
    Use for real-time info, current events, or facts you're unsure about.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            output = []
            for r in results:
                output.append(f"Title: {r['title']}\nSnippet: {r['body']}\n")
            return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"

tools = [calculator, web_search, word_counter]

# ── Create ReAct Agent ──────────────────────────────────────────
# The agent receives a question, thinks about which tool to use,
# calls it, observes results, and decides if it needs more tools.
agent = create_react_agent(
    model,
    tools,
    prompt="You are a helpful research assistant with access to tools. "
           "Use the appropriate tool for each task. "
           "For math, use calculator. For current info, use web_search. "
           "Always explain your reasoning alongside the results."
)

def main():
    print("=" * 60)
    print("  MODULE 4: Tools & Agents")
    print("=" * 60)
    print("\n🤖 Research Agent — I can search, calculate, & analyze!")
    print("   Try: 'What is 245 * 38?'")
    print("   Try: 'Search for latest news about AI'")
    print("   Type 'quit' to exit")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("\n👋 Goodbye!")
            break

        print("\n🤔 Agent is thinking...\n")
        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            messages = result["messages"]
            
            # Show tool usage
            print("📋 Agent Steps:")
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"   🔧 Used: {tc['name']}({tc['args']})")
                elif msg.type == "tool":
                    content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    print(f"   📊 Result: {content}")
            
            final = messages[-1].content
            print(f"\n🤖 Agent: {final}")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("  ✅ Module 4 Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
