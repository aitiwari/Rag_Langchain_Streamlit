from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
class FAISSVDB:
    def __init__(self,main_config) -> None:
        self.main_config = main_config
        
    def faiss_vectorstores_and_retriever(self , docs):
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        faiss_db = FAISS.from_documents(docs, embedding_function)
        faiss_retriever = faiss_db.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )
        
        st.session_state["retriever"] = faiss_retriever
        return faiss_retriever
    
    