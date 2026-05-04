"""
╔══════════════════════════════════════════════════════════════╗
║  MODULE 5a: RAG — Document Ingestion                        ║
╠══════════════════════════════════════════════════════════════╣
║  RAG = Retrieval-Augmented Generation                        ║
║  1. Load documents  2. Split into chunks  3. Create          ║
║  embeddings  4. Store in vector DB                           ║
║                                                              ║
║  RUN THIS FIRST, then run query.py                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
from dotenv import load_dotenv
load_dotenv()

import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    print("=" * 60)
    print("  MODULE 5a: RAG — Document Ingestion Pipeline")
    print("=" * 60)

    # ── Step 1: Load documents ─────────────────────────────────
    print("\n📂 Step 1: Loading documents...")
    docs_path = os.path.join(os.path.dirname(__file__), "sample_docs")
    
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"   ✅ Loaded {len(documents)} document(s)")
    for doc in documents:
        print(f"   📄 {os.path.basename(doc.metadata['source'])}: "
              f"{len(doc.page_content)} chars")

    # ── Step 2: Split into chunks ──────────────────────────────
    # RecursiveCharacterTextSplitter tries paragraph breaks first,
    # then sentences, then words — keeps related info together.
    print("\n✂️  Step 2: Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   ✅ Created {len(chunks)} chunks")
    print(f"\n   📝 Sample chunk #1:")
    print(f"   {'─' * 40}")
    print(f"   {chunks[0].page_content[:200]}...")
    print(f"   {'─' * 40}")

    # ── Step 3: Create embeddings ──────────────────────────────
    # Embeddings convert text → numbers. Similar texts = similar vectors.
    # Using HuggingFace (runs locally, free, no API key needed!)
    print("\n🔢 Step 3: Creating embeddings (local HuggingFace model)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Small, fast, good quality
    )
    
    test_emb = embeddings.embed_query("What is machine learning?")
    print(f"   ✅ Embedding dimension: {len(test_emb)}")
    print(f"   📊 First 5 values: {[round(v, 4) for v in test_emb[:5]]}")

    # ── Step 4: Store in FAISS ─────────────────────────────────
    # FAISS = Facebook AI Similarity Search. Fast vector DB.
    print("\n💾 Step 4: Building FAISS vector store...")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    save_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    vector_store.save_local(save_path)
    print(f"   ✅ Saved to: {save_path}")
    print(f"   📊 Total vectors: {vector_store.index.ntotal}")

    # ── Step 5: Quick test ─────────────────────────────────────
    print("\n🔍 Step 5: Testing retrieval...")
    test_query = "What are the types of machine learning?"
    results = vector_store.similarity_search(test_query, k=3)
    print(f"   Query: '{test_query}'")
    for i, doc in enumerate(results, 1):
        content = doc.page_content[:150] + "..."
        print(f"   📌 Chunk {i}: {content}\n")

    print("=" * 60)
    print("  ✅ Ingestion Complete! Now run query.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
