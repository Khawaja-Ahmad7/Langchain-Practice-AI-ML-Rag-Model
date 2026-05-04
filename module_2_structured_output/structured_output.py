"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 2: Structured Output & Output Parsers               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CONCEPTS YOU'LL LEARN:                                      ║
║  1. Pydantic models        — define the shape of your data   ║
║  2. PydanticOutputParser   — force LLM to return typed data  ║
║  3. Format instructions    — auto-generated parsing rules    ║
║  4. .partial()             — inject static prompt variables   ║
║  5. Why structured output matters for real apps              ║
║                                                              ║
║  THE PROBLEM:                                                ║
║  LLMs return raw text. But your code needs structured data   ║
║  (JSON, objects, typed fields). Output parsers bridge this    ║
║  gap by telling the LLM HOW to format its response, then     ║
║  parsing that response into a Python object.                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
sys.path.insert(0, sys.path[0] + '/..')
sys.stdout.reconfigure(encoding='utf-8')

from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# ── Step 1: Define your data structure with Pydantic ────────────
# Pydantic is Python's standard for data validation.
# Think of it as a "contract" — the LLM MUST return data
# that matches this exact shape, or we get an error.

class ResearchSummary(BaseModel):
    """Structured output for a research topic analysis."""
    
    title: str = Field(description="A concise title for the research topic")
    summary: str = Field(description="A 2-3 sentence summary of the topic")
    key_points: List[str] = Field(description="3-5 key takeaways or facts")
    difficulty_level: str = Field(description="One of: beginner, intermediate, advanced")
    related_topics: List[str] = Field(description="2-3 related topics to explore next")


# ── Step 2: Create the output parser ───────────────────────────
# The parser does TWO things:
#   1. Generates "format instructions" (tells the LLM how to format)
#   2. Parses the LLM's response into a Pydantic object
parser = PydanticOutputParser(pydantic_object=ResearchSummary)

# Let's see what format instructions look like:
print("📋 Format Instructions (sent to the LLM):")
print("-" * 50)
print(parser.get_format_instructions())
print("-" * 50)


# ── Step 3: Create a prompt with format instructions ───────────
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a research assistant. Analyze the given topic and "
     "provide a structured summary.\n\n"
     "You MUST format your response exactly as specified:\n"
     "{format_instructions}"),
    ("human", "Research this topic: {topic}")
])

# .partial() pre-fills a variable so you don't pass it every call
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)


# ── Step 4: Build the chain ────────────────────────────────────
model = get_llm(temperature=0.3)  # Lower temp = more consistent structured output

# Same LCEL pattern: prompt → model → parser
# But now the parser returns a Pydantic object, not a string!
chain = prompt | model | parser


# ── Step 5: Run it! ────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  MODULE 2: Structured Output & Output Parsers")
    print("=" * 60)

    # ─── Demo 1: Get structured output ──────────────────────
    print("\n📘 Demo 1: Structured Research Summary\n")
    
    result = chain.invoke({"topic": "Blockchain Technology"})
    
    # result is now a Pydantic object, NOT a string!
    print(f"📌 Type of result: {type(result).__name__}")
    print(f"\n📖 Title: {result.title}")
    print(f"📝 Summary: {result.summary}")
    print(f"📊 Difficulty: {result.difficulty_level}")
    print(f"\n🔑 Key Points:")
    for i, point in enumerate(result.key_points, 1):
        print(f"   {i}. {point}")
    print(f"\n🔗 Related Topics:")
    for topic in result.related_topics:
        print(f"   → {topic}")

    # ─── Demo 2: Use the structured data programmatically ───
    print("\n\n📘 Demo 2: Using structured data in code\n")
    
    if result.difficulty_level == "beginner":
        print("✅ Great topic for beginners! No prerequisites needed.")
    elif result.difficulty_level == "intermediate":
        print("📚 You should have some foundational knowledge first.")
    else:
        print("🎓 Advanced topic — make sure you understand the basics.")
    
    print(f"\n💡 Next, you could study: {', '.join(result.related_topics)}")

    # ─── Demo 3: Another model — Movie Review ──────────────
    print("\n\n📘 Demo 3: Different Pydantic model — Movie Analysis\n")
    
    class MovieAnalysis(BaseModel):
        """Structured analysis of a movie."""
        movie_name: str = Field(description="The name of the movie")
        genre: str = Field(description="The primary genre")
        rating: float = Field(description="Rating out of 10")
        one_line_review: str = Field(description="A single sentence review")
        watch_if_you_like: List[str] = Field(description="3 similar movies")
    
    movie_parser = PydanticOutputParser(pydantic_object=MovieAnalysis)
    
    movie_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a movie critic. Analyze the movie and return structured data.\n\n"
         "{format_instructions}"),
        ("human", "Analyze this movie: {movie}")
    ]).partial(format_instructions=movie_parser.get_format_instructions())
    
    movie_chain = movie_prompt | model | movie_parser
    
    movie = movie_chain.invoke({"movie": "Inception"})
    
    print(f"🎬 {movie.movie_name} ({movie.genre})")
    print(f"⭐ {movie.rating}/10")
    print(f"📝 {movie.one_line_review}")
    print(f"🎥 Similar: {', '.join(movie.watch_if_you_like)}")

    print("\n" + "=" * 60)
    print("  ✅ Module 2 Complete! You've learned:")
    print("     - Pydantic models for data contracts")
    print("     - PydanticOutputParser for typed responses")
    print("     - Format instructions (auto-generated)")
    print("     - .partial() for pre-filling prompt variables")
    print("     - Why structured output beats raw text")
    print("=" * 60)


if __name__ == "__main__":
    main()
