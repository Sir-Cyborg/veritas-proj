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
    def __init__(self, type, role_prompt=None):
        self.type = type
        self.conversation_id = None
        self.role_prompt = role_prompt or "You are a banking ICT and Security assistant."
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
        

    def augment(self, question, context):
        prompt = f"""{self.role_prompt}.
                    Answer using ONLY the context below.
                    Context: {context}
                    Question: {question}"""
        
        response = self.client.responses.create(model="gpt-4o", input=prompt, conversation=self.conversation_id, store=True)
        return response.output_text
    
    def augment_query(self, question):
        """Augment the question asked to LMM to return query for retriever"""

        prompt = f"""Given the question below, generate ONLY a search query 
                    to find relevant information in a vector database.
                    Question: {question}"""
        query = self.client.responses.create(model="gpt-4o", input=prompt, conversation=self.conversation_id, store=True)
        return query.output_text
    
class Judger:
    def __init__(self, client):
        self.client = client

    def judge(self, question, context, answers):
        formatted_answers = "\n\n".join(
        f"{name}:\n{answer}" for name, answer in answers
    )
        prompt = f"""You are a senior banking ICT and security reviewer.

                        Question:
                        {question}

                        Context:
                        {context}

                        Below are answers from multiple experts:
                        {formatted_answers}

                        Task:
                        - Compare the answers
                        - Resolve contradictions
                        - Produce a single, accurate, well-justified final answer
                        - Use ONLY the provided context"""
        
        response = self.client.responses.create(model="gpt-4o", input=prompt, store=True)
        return response.output_text
    
def run_court(question, context, augmentors):
    """Run multiple augmentors (ensamble) as experts to answer the question."""
    answers = []
    for i, augmentor in enumerate(augmentors, start=1):
        answer = augmentor.augment(question=question, context=context)
        answers.append((f"Expert {i}", answer))
    return answers


def chat():
    retriever = Retriever()

    query_augmentor = Augmentor(type="openai")
    augmentors = [
        Augmentor(type="openai", role_prompt="You are a conservative banking compliance expert."),
        Augmentor("openai", role_prompt="You are a pragmatic ICT operations engineer."),
        Augmentor("openai", role_prompt="You are a critical security auditor.")
    ]

    judger = Judger(augmentors[0].client)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        query = query_augmentor.augment_query(question)
        context = retriever.retrieve(query=query)

        answers = run_court(question=question, context=context, augmentors=augmentors)
        verdict = judger.judge(question=question, context=context, answers=answers)

        print("\nFinal Answer:\n", verdict)
        print("\nContext:\n", context)
        print("\nAnswers:\n", answers)

        print("\n" + "="*50 + "\n")
    

if __name__ == "__main__":
    chat()