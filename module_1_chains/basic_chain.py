"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 1: Chains & LCEL (LangChain Expression Language)     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CONCEPTS YOU'LL LEARN:                                      ║
║  1. ChatOpenAI              — the LLM wrapper (via OpenRouter)║
║  2. ChatPromptTemplate      — reusable prompt blueprints     ║
║  3. StrOutputParser          — extracts text from LLM reply  ║
║  4. LCEL pipe operator (|)  — chains components together     ║
║  5. .invoke()               — run the chain once             ║
║  6. .stream()               — get tokens as they generate    ║
║  7. .batch()                — process multiple inputs        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Step 1: Setup ───────────────────────────────────────────────
import sys
sys.path.insert(0, sys.path[0] + '/..')  # allow importing config.py
sys.stdout.reconfigure(encoding='utf-8')

from config import get_llm


# ── Step 2: Import LangChain components ─────────────────────────
# These are the 3 building blocks of every chain:
#   Prompt  →  Model  →  Output Parser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── Step 3: Initialize the LLM (Language Model) ────────────────
# get_llm() is defined in config.py — change the model ONCE there
# and every module picks it up. No more editing 5 files!
model = get_llm()


# ── Step 4: Create a Prompt Template ───────────────────────────
# A template is like a fill-in-the-blank form.
# {topic} is a variable that gets replaced at runtime.
#
# ChatPromptTemplate.from_messages() lets you define the
# conversation structure:
#   - "system": instructions for the AI's behavior
#   - "human": the user's message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful teacher who explains concepts clearly and concisely. "
               "Use simple language and give a real-world analogy when possible."),
    ("human", "Explain the following topic in 3-4 sentences: {topic}")
])


# ── Step 5: Create an Output Parser ───────────────────────────
# The LLM returns an AIMessage object. The parser extracts
# just the text string from it — much easier to work with.
parser = StrOutputParser()


# ── Step 6: Build the Chain using LCEL ─────────────────────────
# This is the magic of LCEL! The pipe operator (|) connects
# components in sequence:
#
#   prompt  →  model  →  parser
#   (format)   (think)   (extract text)
#
# Under the hood, each component implements the "Runnable"
# interface, so they can be chained together seamlessly.
chain = prompt | model | parser


# ── Step 7: Run the chain! ─────────────────────────────────────
def main():
    print("=" * 60)
    print("  MODULE 1: Chains & LCEL — LangChain Basics")
    print("=" * 60)

    # ─── Demo 1: .invoke() — Single execution ──────────────
    # .invoke() runs the chain once with the given input.
    # The input is a dictionary matching the template variables.
    print("\n📘 Demo 1: .invoke() — Single call\n")
    
    result = chain.invoke({"topic": "What is an API?"})
    print(f"Result: {result}")

    # ─── Demo 2: .stream() — Token-by-token streaming ──────
    # .stream() yields tokens as they're generated.
    # Great for real-time UIs where you want text to appear
    # word-by-word (like ChatGPT's typing effect).
    print("\n\n📘 Demo 2: .stream() — Streaming response\n")
    print("Result: ", end="")
    
    for chunk in chain.stream({"topic": "What is Machine Learning?"}):
        print(chunk, end="", flush=True)
    print()  # newline after streaming

    # ─── Demo 3: .batch() — Multiple inputs at once ────────
    # .batch() processes a list of inputs in parallel.
    # Much faster than calling .invoke() in a loop!
    print("\n\n📘 Demo 3: .batch() — Multiple topics at once\n")
    
    topics = [
        {"topic": "What is a database?"},
        {"topic": "What is cloud computing?"},
        {"topic": "What is version control?"},
    ]
    
    results = chain.batch(topics)
    
    for topic, result in zip(topics, results):
        print(f"📌 {topic['topic']}")
        print(f"   {result}\n")

    # ─── Demo 4: Swap the prompt on the fly ─────────────────
    # Since LCEL is modular, you can create a NEW chain
    # by swapping just one component. The model and parser
    # stay the same!
    print("\n📘 Demo 4: Swapping prompts — Same model, different behavior\n")
    
    joke_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a witty comedian. Give short, clever jokes."),
        ("human", "Tell me a joke about: {topic}")
    ])
    
    # New chain, same model and parser!
    joke_chain = joke_prompt | model | parser
    
    joke = joke_chain.invoke({"topic": "programming"})
    print(f"😄 {joke}")

    print("\n" + "=" * 60)
    print("  ✅ Module 1 Complete! You've learned:")
    print("     - ChatPromptTemplate (prompt blueprints)")
    print("     - ChatOpenAI via OpenRouter (LLM gateway)")
    print("     - StrOutputParser (text extraction)")
    print("     - LCEL pipe operator (chaining)")
    print("     - .invoke(), .stream(), .batch()")
    print("=" * 60)


if __name__ == "__main__":
    main()
