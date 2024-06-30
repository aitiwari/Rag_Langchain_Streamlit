from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
class Neo4jGDB:
    def __init__(self,main_config) -> None:
        self.main_config = main_config
        
    def neo4j_vectorstores_and_retriever(self , docs):
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # load it into Chroma
        #index_name = st.session_state["neo4j_index"]
        neo4j_db = Neo4jVector.from_documents(docs, embedding=embedding_function)
        neo4j_retriever = neo4j_db.as_retriever(
            #search_type="hybrid",  # Also test "similarity"
            search_kwargs={"k": 8},
        )
        
        st.session_state["retriever"] = neo4j_retriever
        return neo4j_retriever
    
    