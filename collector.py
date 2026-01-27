from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

def chroma_collection():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="policy_procedures",
        embedding_function=embedding_function
    )
    return collection


def load_chunks_to_collection(chunks_dir, collection):
    """
    Iterates over .txt chunk files and adds them to a Chroma collection.

    - id            -> chunk id (e.g. chunk_001)
    - document      -> chunk text WITHOUT section number (e.g. without '1.')
    - metadata:
        * original_file
        * chunk_id
        * chunk_file
        * section_number
        * section_title
    """

    chunks_dir = Path(chunks_dir)

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    for chunk_path in sorted(chunks_dir.glob("*.txt")):

        # -------------------------
        # 1. Parse filename
        # -------------------------
        # Example:
        # Access_Management_Operational_Procedure__chunk_001__0_full_document.txt

        filename = chunk_path.name
        name_parts = chunk_path.stem.split("__")

        original_file = f"{name_parts[0]}.txt"
        chunk_id = name_parts[1] if len(name_parts) > 1 else "chunk_unknown"

        # -------------------------
        # 2. Read file content
        # -------------------------
        text = chunk_path.read_text(encoding="utf-8", errors="replace").strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # -------------------------
        # 3. Extract section info
        # -------------------------
        # Expected first line:
        # "1. Introduction"

        section_number = "0"
        section_title = "FULL_DOCUMENT"
        document_text = text

        if lines:
            first_line = lines[0]
            parts = first_line.split(maxsplit=1)

            # Check if first token is a number like "1." or "5.3"
            token = parts[0].rstrip(".")
            if all(p.isdigit() for p in token.split(".")):
                section_number = parts[0]          # keep "1."
                section_title = parts[1] if len(parts) > 1 else ""
                # document text = everything AFTER the section line
                document_text = "\n".join(lines[1:]).strip()

        # -------------------------
        # 4. Add to collection
        # -------------------------
        collection.add(
            ids=[chunk_id],
            documents=[document_text],
            metadatas=[{
                "original_file": original_file,
                "chunk_id": chunk_id,
                "chunk_file": filename,
                "section_number": section_number,
                "section_title": section_title,
            }]
        )

        print(f"Added chunk: {chunk_id} ({filename})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load text chunks into ChromaDB collection.")
    parser.add_argument(
        "--chunks_dir",
        type=str,
        required=True,
        help="Path to the directory containing text chunk files."
    )

    args = parser.parse_args()
    
    collection = chroma_collection()
    
    load_chunks_to_collection(args.chunks_dir, collection)