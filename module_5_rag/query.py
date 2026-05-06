"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 5b: RAG — Question Answering                         ║
╠══════════════════════════════════════════════════════════════╣
║  Loads the vector store, builds a retrieval chain, and       ║
║  answers questions grounded in YOUR documents.               ║
║                                                              ║
║  PREREQUISITE: Run ingest.py first!                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys
sys.path.insert(0, sys.path[0] + '/..')
sys.stdout.reconfigure(encoding='utf-8')

from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def format_docs(docs):
    """Combine retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def main():
    print("=" * 60)
    print("  MODULE 5b: RAG — Document Q&A")
    print("=" * 60)

    # ── Load vector store ──────────────────────────────────────
    print("\n💾 Loading vector store...")
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    if not os.path.exists(index_path):
        print("   ❌ Not found! Run ingest.py first.")
        return
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"   ✅ Loaded {vector_store.index.ntotal} vectors")

    # ── Create retriever ───────────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # ── RAG prompt ─────────────────────────────────────────────
    # Tells the LLM to answer ONLY from the provided context.
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant. Answer ONLY from the context below. "
         "If the context doesn't have the answer, say so."),
        ("human", 
         "Context:\n{context}\n\nQuestion: {question}")
    ])

    # ── Build RAG chain ────────────────────────────────────────
    model = get_llm(temperature=0.3)
    
    # LCEL RAG chain: retrieve context + pass question → prompt → model
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )
    print("   🔗 RAG chain ready!")

    # ── Interactive Q&A ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("📚 Ask about AI & Machine Learning!")
    print("   Try: 'What are the types of AI?'")
    print("   Try: 'Explain deep learning'")
    print("   'sources' → see raw retrieval | 'quit' → exit")
    print("-" * 60)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "quit":
            print("\n👋 Goodbye!")
            break
        if question.lower() == "sources":
            q = input("   Query: ").strip()
            if q:
                docs = retriever.invoke(q)
                for i, doc in enumerate(docs, 1):
                    print(f"\n   ── Chunk {i} ──")
                    print(f"   {doc.page_content[:200]}...")
            continue

        print("\n🤖 ", end="", flush=True)
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print()

    print("\n" + "=" * 60)
    print("  ✅ Module 5 Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
