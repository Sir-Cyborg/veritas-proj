import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", collection="policy_procedures"):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection,
            embedding_function=embedder
        )

    def retrieve(self, query, k=3):
        results = self.collection.query(
            query_texts=[query],
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

        return context
    
class Augmentor:
    def __init__(self, type):
        self.type = type
        self.conversation_id = None
        self.setup()


    def setup(self):
        if self.type == "openai":
            load_dotenv()
            api_key = os.getenv("OPENAI_API")
            if not api_key:
                raise RuntimeError("OPENAI_API not found")
            self.client = OpenAI(api_key=api_key)
            self.conversation_id = self.client.conversations.create().id
            pass
        if self.type == "local":
            # setup local model client
            pass
        

    def augment(self, query, context):
        prompt = f"""You are a banking ICT and security assistant.
                    Answer using ONLY the context below.
                    Context: {context}
                    Question: {query}"""
        
        response = self.client.responses.create(model="gpt-4o", input=prompt, conversation=self.conversation_id, store=True)
        return response.output_text

def chat():
    retriever = Retriever()
    augmentor = Augmentor(type="openai")

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        context = retriever.retrieve(query=query)
        answer = augmentor.augment(query=query, context=context)

        print("\nAnswer:\n", answer)
        print("\n" + "="*50 + "\n")
    

if __name__ == "__main__":
    chat()