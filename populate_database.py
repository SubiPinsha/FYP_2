

# import argparse
# import os
# import shutil
# from pypdf import PdfReader
# from pypdf.errors import PdfReadError
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from get_embedding_function import get_embedding_function

# DATA_DIR = "data"
# CHROMA_DIR = "chroma"

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true")
#     args = parser.parse_args()

#     if args.reset:
#         clear_database()

#     index_language("tamil")
#     index_language("english")

# def index_language(lang: str):
#     data_path = os.path.join(DATA_DIR, lang)
#     chroma_path = os.path.join(CHROMA_DIR, lang)

#     print(f"\n📘 Indexing {lang.upper()} documents")
#     documents = load_documents(data_path)
#     chunks = split_documents(documents)
#     add_to_chroma(chunks, chroma_path)

# def load_documents(path: str):
#     documents: list[Document] = []
#     if not os.path.exists(path):
#         print(f"⚠️  Data path does not exist: {path}")
#         return documents

#     for fname in sorted(os.listdir(path)):
#         if not fname.lower().endswith(".pdf"):
#             continue

#         file_path = os.path.join(path, fname)
#         try:
#             reader = PdfReader(file_path)
#             for i, page in enumerate(reader.pages, start=1):
#                 text = page.extract_text() or ""
#                 metadata = {"source": fname, "page": i}
#                 documents.append(Document(page_content=text, metadata=metadata))
#         except PdfReadError as e:
#             print(f"⚠️ Skipping '{fname}': PdfReadError: {e}")
#         except Exception as e:
#             print(f"⚠️ Skipping '{fname}': {type(e).__name__}: {e}")

#     return documents

# def split_documents(documents: list[Document]):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#     )
#     return splitter.split_documents(documents)

# def add_to_chroma(chunks: list[Document], chroma_path: str):
#     db = Chroma(
#         persist_directory=chroma_path,
#         embedding_function=get_embedding_function()
#     )

#     chunks = calculate_chunk_ids(chunks)
#     existing_ids = set(db.get(include=[])["ids"])

#     new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]

#     if new_chunks:
#         print(f"👉 Adding {len(new_chunks)} chunks")
#         db.add_documents(
#             new_chunks,
#             ids=[c.metadata["id"] for c in new_chunks]
#         )
#     else:
#         print("✅ No new documents")

# def calculate_chunk_ids(chunks):
#     last_page_id = None
#     index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         page_id = f"{source}:{page}"

#         if page_id == last_page_id:
#             index += 1
#         else:
#             index = 0

#         chunk.metadata["id"] = f"{page_id}:{index}"
#         last_page_id = page_id

#     return chunks

# def clear_database():
#     if os.path.exists(CHROMA_DIR):
#         shutil.rmtree(CHROMA_DIR)
#         print("🗑️ Cleared all databases")

# if __name__ == "__main__":
#     main()

import argparse
import os
import shutil
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

DATA_DIR = "data"
CHROMA_DIR = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--lang", choices=["tamil", "english"], required=True, help="Language to index.")
    args = parser.parse_args()

    if args.reset:
        clear_database(args.lang)

    # Index the specific language provided in the command
    index_language(args.lang)

def index_language(lang: str):
    data_path = os.path.join(DATA_DIR, lang)
    chroma_path = os.path.join(CHROMA_DIR, lang)

    print(f"\n📘 Indexing {lang.upper()} documents from: {data_path}")
    documents = load_documents(data_path)
    if not documents:
        print(f"⚠️ No documents found for {lang}.")
        return

    chunks = split_documents(documents)
    add_to_chroma(chunks, chroma_path, lang)

import fitz  # This is the PyMuPDF library

def load_documents(path: str):
    documents: list[Document] = []
    if not os.path.exists(path):
        print(f"⚠️ Data path does not exist: {path}")
        return documents

    for fname in sorted(os.listdir(path)):
        if not fname.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(path, fname)
        try:
            # fitz.open is much more robust against corrupted elementary objects
            doc = fitz.open(file_path)
            print(f"📄 Opening '{fname}' - Found {len(doc)} total pages.")
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                
                # Only add if the page actually has content
                if text:
                    metadata = {"source": fname, "page": page_num}
                    documents.append(Document(page_content=text, metadata=metadata))
                else:
                    print(f"   ℹ️ Skipping empty page {page_num}")
            
            doc.close()
        except Exception as e:
            print(f"⚠️ Failed to read '{fname}': {e}")

    print(f"✅ Successfully loaded {len(documents)} pages into memory.")
    return documents

def split_documents(documents: list[Document]):
    # Note: 800/80 is a good balance for textbooks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks.")
    return chunks

def add_to_chroma(chunks: list[Document], chroma_path: str, lang: str):
    # Initialize Chroma with the specific embedding function for the language
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function(lang)
    )

    # Calculate unique IDs so we don't add the same content twice
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Get existing IDs from the DB
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"🔢 Number of existing documents in {lang} DB: {len(existing_ids)}")

    # Only add chunks that aren't already in the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} new chunks to {lang} database...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ Database is already up to date.")

def calculate_chunk_ids(chunks):
    """
    Creates IDs like 'filename.pdf:page_num:chunk_index'
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the final chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page metadata
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database(lang: str):
    path = os.path.join(CHROMA_DIR, lang)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"🗑️ Cleared {lang} database at {path}")
    else:
        print(f"ℹ️ No database found to clear at {path}")

if __name__ == "__main__":
    main()