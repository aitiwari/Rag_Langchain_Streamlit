import os
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.vectorstores import Weaviate

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
class WeaviateVDB:
    def __init__(self,main_config) -> None:
        self.main_config = main_config
        
    def weaviate_vectorstores_and_retriever(self , docs):
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # load it into Weaviate
        WEAVIATE_API_KEY=os.environ['WEAVIATE_API_KEY']
        WEAVIATE_URL = os.environ['WEAVIATE_URL'] 
        #weaviate_client = weaviate.connect_to_local()
        weaviate_client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY))
        weaviate_conn = Weaviate(weaviate_client, "Langchain", "text_key")
        
        weaviate_db = weaviate_conn.from_documents(docs, embedding_function)
        weaviate_retriever = weaviate_db.as_retriever(
            #search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )
        
        st.session_state["retriever"] = weaviate_retriever
        return weaviate_retriever
    
    