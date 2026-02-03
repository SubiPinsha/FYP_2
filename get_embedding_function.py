# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings


# def get_embedding_function():
#     # embeddings = BedrockEmbeddings(
#     #     credentials_profile_name="default", region_name="us-east-1"
#     # )
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings

from langchain_ollama import OllamaEmbeddings

def get_embedding_function(lang: str):
    if lang == "tamil":
        # Better for the complex structure of Tamil
        return OllamaEmbeddings(model="bge-m3")
    # Default for English
    return OllamaEmbeddings(model="nomic-embed-text")