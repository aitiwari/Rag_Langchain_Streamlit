from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
class ChromaVDB:
    def __init__(self,main_config) -> None:
        self.main_config = main_config
        
    def chroma_vectorstores_and_retriever(self , docs):
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        chroma_db = Chroma.from_documents(docs, embedding_function)
        chroma_retriever = chroma_db.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )
        
        st.session_state["retriever"] = chroma_retriever
        return chroma_retriever
    
    