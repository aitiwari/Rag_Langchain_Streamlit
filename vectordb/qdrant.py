from langchain_community.document_loaders import TextLoader
#from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter


class QdrantVDB:
    def __init__(self,main_config,repo_path,groq_api_key,selected_model, rag_path_ext,query) -> None:
        self.main_config = main_config
        self.repo_path = repo_path
        self.groq_api_key = groq_api_key
        self.selected_model = selected_model
        self.rag_path_ext= rag_path_ext
        self.query = query

    def vectorstores_and_get_context_retriever(self,docs):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # print("********************docs*************")
        # print(docs)
        qdrant = Qdrant.from_documents(
            docs,
            embeddings,
            location=":memory:",  # Local mode with in-memory storage only
            collection_name="my_documents",
        )

        
        # rel_context_docs = qdrant.similarity_search(self.query)
        # print(rel_context_docs)

        retriever = qdrant.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )
        return retriever
        


        

