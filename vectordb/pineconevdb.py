from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import streamlit as st
class PineconeVDB:
    def __init__(self,main_config) -> None:
        self.main_config = main_config
        
    def pinecone_vectorstores_and_retriever(self , docs):
        try :
            # create the open-source embedding function
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            # load it into Pinecone
            index_name = st.session_state["pinecone_index"]
            pinecone_db = PineconeVectorStore.from_documents(docs,index_name=index_name, embedding=embedding_function)
            pinecone_retriever = pinecone_db.as_retriever(
                #search_type="mmr",  # Also test "similarity"
                search_kwargs={"k": 8},
            )
            
            st.session_state["retriever"] = pinecone_retriever
            
        except Exception as e:
            raise ValueError(f"Error Occured in pinecone storing. Error is {e}")
        return pinecone_retriever
        
    