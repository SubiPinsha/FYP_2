
# import argparse
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama
# from get_embedding_function import get_embedding_function

# PROMPT_TEMPLATE = """
# Answer strictly in {language}.
# Answer ONLY using the context below.

# Context:
# {context}

# Question:
# {question}
# """

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="User query")
#     parser.add_argument(
#         "--lang",
#         choices=["tamil", "english"],
#         required=True,
#         help="Retrieval & output language"
#     )
#     args = parser.parse_args()

#     query_rag(args.query_text, args.lang)

# def query_rag(query_text: str, lang: str):
#     chroma_path = f"chroma/{lang}"

#     db = Chroma(
#         persist_directory=chroma_path,
#         embedding_function=get_embedding_function()
#     )

#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join(
#         [doc.page_content for doc, _ in results]
#     )

#     prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
#         context=context_text,
#         question=query_text,
#         language=lang
#     )

#     model = Ollama(model="llama3.2")
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id") for doc, _ in results]

#     print(f"\nResponse:\n{response_text}\n")
#     print(f"Sources:\n{sources}")

#     return response_text

# if __name__ == "__main__":
#     main()

import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_embedding_function import get_embedding_function

# --- PROMPTS ---

# Template for the final answer
PROMPT_TEMPLATE = """
Answer strictly in {language}.
Answer ONLY using the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

# Template for generating Tamil query variations (Multi-Query)
QUERY_PROMPT_TAMIL = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant proficient in Tamil.
    The user is studying from a textbook. Generate five different versions of the given 
    user question in Tamil to help retrieve the most relevant sections from the database.
    Original question: {question}""",
)

# --- CORE FUNCTIONS ---

def normalize_query(query_text: str, target_lang: str, llm):
    """Translates or fixes the query to match the target database language."""
    normalization_prompt = f"""
    You are a bilingual education assistant. 
    Convert the following user query into {target_lang}. 
    If it's already in {target_lang} or is Tanglish, convert it to proper {target_lang} script.
    User Query: {query_text}
    Result (output only the text):"""
    
    response = llm.invoke(normalization_prompt)
    return response.strip()

def query_rag(query_text: str, lang: str):
    # 1. Initialize Database & LLM
    chroma_path = f"chroma/{lang}"
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=get_embedding_function()
    )
    llm = Ollama(model="llama3.2")

    # 2. Language Normalization (Handling queries in any language)
    print(f"\n[1/3] Normalizing query for {lang} database...")
    search_query = normalize_query(query_text, lang, llm)
    print(f"      Optimized Query: {search_query}")

    # 3. Retrieval Step
    print(f"[2/3] Retrieving relevant context...")
    if lang == "tamil":
        # Using Multi-Query for better Tamil retrieval
        retriever = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(search_kwargs={"k": 3}), 
            llm=llm,
            prompt=QUERY_PROMPT_TAMIL
        )
        results = retriever.invoke(search_query)
    else:
        # Standard search for English
        results = db.similarity_search(search_query, k=5)

    # Combine document contents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # 4. Final Generation
    print(f"[3/3] Generating final response in {lang}...")
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        question=search_query,
        language=lang
    )

    response_text = llm.invoke(prompt)

    # Extract source metadata (IDs)
    sources = [doc.metadata.get("id", "Unknown") for doc in results]

    print(f"\nRESPONSE:\n{response_text}")
    print(f"\nSOURCES: {sources}")
    
    return response_text

# --- CLI ENTRY POINT ---

def main():
    parser = argparse.ArgumentParser(description="Bilingual RAG Query Script")
    parser.add_argument("query_text", type=str, help="The question you want to ask.")
    parser.add_argument(
        "--lang", 
        choices=["tamil", "english"], 
        required=True, 
        help="The language of the textbook/database you want to search."
    )
    
    args = parser.parse_args()
    query_rag(args.query_text, args.lang)

if __name__ == "__main__":
    main()
