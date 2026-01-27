import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

def setup_rag():
    load_dotenv()

    api_key = os.getenv("OPENAI_API")
    if not api_key:
        raise RuntimeError("OPENAI_API not found")

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(
        name="policy_procedures",
        embedding_function=embedder
    )

    client = OpenAI(api_key=api_key)

    return collection, client

def rag_answer(collection, client, question, k=3):
    results = collection.query(
        query_texts=[question],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(docs, metas):
        header = (
            f"Source: {meta['original_file']} | "
            f"Section {meta['section_number']} {meta['section_title']}"
        )
        context_blocks.append(f"{header}\n{doc}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a banking ICT and security assistant.
Answer using ONLY the context below.


Context:
{context}

Question:
{question}

Answer:
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text
def chat():
    collection, client = setup_rag()

    print("RAG chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye 👋")
            break

        answer = rag_answer(collection, client, user_input)
        print("\nRAG:", answer, "\n")

if __name__ == "__main__":
    chat()
