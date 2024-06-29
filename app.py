from langchain_text_splitters import TextSplitter
import streamlit as st
import pandas as pd

from Agents.CSVAgent import CSVAgent
from LLM.groq import Groq
from Loaders.csvloader import CSVLoaders
from Loaders.generic import  GenericLoaders
from Loaders.pdfloader import PDFLoader
from Splitters.codesplitters import CodeSplitters
from Splitters.textspiltters import RecursiveSplitters
from vectordb.qdrant import QdrantVDB
import configparser

# Placeholder for Groq API key (replace with your actual key)

#DEFAULT SELECTION PARAMS - START 
db_option = "Qdrant"
#repo part - can be taken from ui , as of now pointing to local
repo_path = "./Rag_Documents/"
groq_api_key = "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"
selected_model = "mixtral-8x7b-32768"
rag_path_ext = ".java"
config_file_path = "./config.ini"
#DEFAULT SELECTION PARAMS - END 


#config
main_config = configparser.ConfigParser()
main_config.read(config_file_path)

#Process Code Repo 
def process_code_repo(main_config,db_option,repo_path,groq_api_key,selected_model, rag_path_ext,query):
    if db_option == "Qdrant":
        # LOAD
        obj_loaders = GenericLoaders(main_config,repo_path, rag_path_ext)
        documents = obj_loaders.loaders()

        # TRANSFORM
        obj_code_splitters = CodeSplitters(main_config,repo_path,rag_path_ext)
        docs = obj_code_splitters.code_splitters(documents)

        #EMBEDDINGS and stored into VDB
        obj_qdr = QdrantVDB(main_config,repo_path,groq_api_key,selected_model, rag_path_ext,query)

        retriever = obj_qdr.vectorstores_and_get_context_retriever(docs)

        obj_llm_invoke = Groq(groq_api_key,selected_model)

        response = obj_llm_invoke.invoke_llm_with_history(retriever,query)

        return response

        
        
    elif db_option == "Chroma":
        # Implement Chroma-specific query execution logic here
        # (e.g., using the Chroma Python library)
        result = pd.DataFrame({"chroma_data": ["Placeholder for Chroma results"]})
    else:
        st.error("Invalid database option selected.")
        result = None
    return result


# Process PDF files
def process_pdf_files(main_config,db_option,repo_path,groq_api_key,selected_model, rag_path_ext,query):
    if db_option == "Qdrant":
        # LOAD
        obj_loaders = PDFLoader(repo_path, rag_path_ext)
        documents = obj_loaders.load_and_process_pdfs()

        # TRANSFORM
        obj_text_splitters = RecursiveSplitters()
        docs = obj_text_splitters.text_splitters(documents)

        #EMBEDDINGS and stored into VDB
        obj_qdr = QdrantVDB(main_config,repo_path,groq_api_key,selected_model, rag_path_ext,query)

        retriever = obj_qdr.vectorstores_and_get_context_retriever(docs)

        obj_llm_invoke = Groq(groq_api_key,selected_model)

        response = obj_llm_invoke.invoke_llm_with_history(retriever,query)

        return response

        
        
    elif db_option == "Chroma":
        # Implement Chroma-specific query execution logic here
        # (e.g., using the Chroma Python library)
        result = pd.DataFrame({"chroma_data": ["Placeholder for Chroma results"]})
    else:
        st.error("Invalid database option selected.")
        result = None
    return result

# Process CSV Files
def process_csv_files(uploaded_files,query):
    obj_llm = Groq(groq_api_key,selected_model)
    llm= obj_llm.get_llm()
    #***************************
    #agent approach - TODO
    # obj_agent = CSVAgent(llm)
    # agent = obj_agent.create_agent(uploaded_files)
    # result = obj_agent.run_agent(agent,query)
    #***************************
    
    # LOAD
    obj_loaders = CSVLoaders(uploaded_files)
    documents = obj_loaders.csv_loader()

    #EMBEDDINGS and stored into VDB
    obj_qdr = QdrantVDB(main_config,repo_path,groq_api_key,selected_model, rag_path_ext,query)

    retriever = obj_qdr.vectorstores_and_get_context_retriever(documents)

    obj_llm_invoke = Groq(groq_api_key,selected_model)

    response = obj_llm_invoke.invoke_llm_with_history(retriever,query)
    return response



st.set_page_config(page_title="🦜LangChain - Q&A with RAG ", layout="wide")
st.header("🦜LangChain - Q&A with RAG ")


st.sidebar.header("Vector Stores Options")
db_option = st.sidebar.selectbox("Select Vector DB:", ["Qdrant", "Chroma"])

st.sidebar.header("RAG Document Format")
file_format = st.sidebar.selectbox("Select Input Document Format:", ["Code Repo", "PDF", "CSV"])


if file_format == "Code Repo":
    st.sidebar.header("Select Language")
    lang_options = ["java", "python", "csharp"]  # Replace with your actual models
    selected_lang = st.sidebar.selectbox("Select Model:", lang_options)
    if selected_lang == "java":
        rag_path_ext = ".java"
    elif selected_lang == "python":
        rag_path_ext = ".py"
    elif selected_lang == "csharp":
        rag_path_ext = ".cs"

elif file_format == "PDF":
    rag_path_ext = ".pdf"

elif file_format == "CSV":
    rag_path_ext = ".csv"


uploaded_files = st.file_uploader(f"Upload a {rag_path_ext} file", type=rag_path_ext,accept_multiple_files=True,)


st.sidebar.header("Groq API Key")
groq_api_key_input = st.sidebar.text_input("Enter Groq API Key (if applicable)", type="password")
if groq_api_key_input:
    groq_api_key = groq_api_key_input  # Update placeholder with user-provided key
    st.sidebar.header("Model Selection (if applicable)")
    model_options = ["mixtral-8x7b-32768","llama3-8b-8192", "llama3-70b-8192"]  # Replace with your actual models
    selected_model = st.sidebar.selectbox("Select Model:", model_options)

query = st.chat_input("Enter your query here")

if query:
    st.write(f"🧑‍💻: {query}")
    try : 
        result = ""
        if file_format == "Code Repo":
            if len(uploaded_files) > 0 :
                repo_path = uploaded_files
            
            result = process_code_repo(main_config,db_option,repo_path,groq_api_key,selected_model, rag_path_ext,query)

        elif file_format == "PDF":
            if len(uploaded_files) > 0 :
                repo_path = uploaded_files
            result = process_pdf_files(main_config,db_option,repo_path,groq_api_key,selected_model, rag_path_ext,query)
        elif file_format == "CSV":
            result = process_csv_files(uploaded_files,query)
        else:
            st.error("Invalid format selected.")

        if result is not None:
            st.write(f"🤖:{result}")
            
        else:
            st.warning("Response is Empty")

    except Exception as e:
        st.warning(f"Internal Server Error : {e}")

st.sidebar.info("**Note:**")
st.sidebar.write(
    "- Replace placeholder groq API KEY by creating account https://console.groq.com/keys and select the model - default is mixtral"
    "- Chroma Db storage in local and can be view the chunk storage "
    "- Qdrant is used as In-Memory storage"
    "- Code repo file - https://github.com/Priyansh42/Lung-Cancer-Detection/blob/main/lcd_cnn.py"
)
