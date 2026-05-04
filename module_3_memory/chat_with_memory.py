"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 3: Conversational Memory                            ║
╠══════════════════════════════════════════════════════════════╣
║  LLMs are STATELESS. Memory solves this by storing previous  ║
║  messages and replaying them with each new request.          ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
sys.path.insert(0, sys.path[0] + '/..')
sys.stdout.reconfigure(encoding='utf-8')

from config import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── LLM (from shared config.py) ────────────────────────────────
model = get_llm()

# ── Prompt with history placeholder ─────────────────────────────
# MessagesPlaceholder("history") is where conversation history
# gets injected automatically by RunnableWithMessageHistory.
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a friendly study buddy and learning assistant. "
     "You help users learn new topics by explaining concepts, "
     "giving examples, and asking follow-up questions. "
     "Keep responses concise. Remember details the user shares."),
    MessagesPlaceholder("history"),  # ← conversation history goes here
    ("human", "{input}")
])

chain = prompt | model | StrOutputParser()

# ── Memory store (keyed by session_id) ─────────────────────────
# Multiple sessions = multiple independent conversations
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ── Wrap chain with memory ─────────────────────────────────────
# This transforms a stateless chain into a stateful one!
# It auto-loads history, injects it, runs the chain, then saves.
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def main():
    print("=" * 60)
    print("  MODULE 3: Conversational Memory")
    print("=" * 60)
    print("\n🧠 Study Buddy Chat — I remember our conversation!")
    print("   'new' → new session | 'history' → view msgs | 'quit' → exit")
    print("-" * 60)

    session_id = "session_1"
    print(f"\n📌 Current session: {session_id}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("\n👋 Goodbye! Keep learning!")
            break
        if user_input.lower() == "new":
            session_id = f"session_{len(store) + 1}"
            print(f"\n🔄 Started new session: {session_id}\n")
            continue
        if user_input.lower() == "history":
            history = get_session_history(session_id)
            print(f"\n📜 Messages in '{session_id}':")
            for msg in history.messages:
                role = "You" if msg.type == "human" else "AI"
                content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                print(f"   [{role}]: {content}")
            print()
            continue

        config = {"configurable": {"session_id": session_id}}
        print("AI: ", end="", flush=True)
        for chunk in chain_with_memory.stream({"input": user_input}, config=config):
            print(chunk, end="", flush=True)
        print("\n")

    print("\n" + "=" * 60)
    print(f"  📊 Sessions: {len(store)}")
    for sid, h in store.items():
        print(f"     {sid}: {len(h.messages)} messages")
    print("  ✅ Module 3 Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
