from typing import Sequence
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
from vectordb.chroma import ChromaVDB
from vectordb.qdrant import QdrantVDB
import configparser

# Placeholder for Groq API key (replace with your actual key)

#DEFAULT SELECTION PARAMS - START 
selected_db = "Qdrant"
#repo part - can be taken from ui , as of now pointing to local
repo_path = "./Rag_Documents/"
groq_api_key = "gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
selected_model = "mixtral-8x7b-32768"
rag_path_ext = ".java"
config_file_path = "./config.ini"
#DEFAULT SELECTION PARAMS - END 


#config
main_config = configparser.ConfigParser()
main_config.read(config_file_path)

#Process Code Repo 
def process_code_repo(main_config,selected_db,repo_path,groq_api_key,selected_model, rag_path_ext,query):
    
    #LOAD and SPLIT
    obj_loaders_splitter = GenericLoaders(main_config,repo_path, rag_path_ext)
    docs = obj_loaders_splitter.code_loader_and_splitters()
    if selected_db == "Qdrant":
        # # LOAD
        # obj_loaders = GenericLoaders(main_config,repo_path, rag_path_ext)
        # documents = obj_loaders.loaders()

        # # TRANSFORM
        # obj_code_splitters = CodeSplitters(main_config,repo_path,rag_path_ext)
        # docs = obj_code_splitters.code_splitters(documents)
        
        

        #EMBEDDINGS and stored into VDB
        obj_qdr = QdrantVDB(main_config,repo_path,groq_api_key,selected_model, rag_path_ext,query)

        retriever = obj_qdr.vectorstores_and_get_context_retriever(docs)
        
        st.session_state["retriever"] = retriever

        # obj_llm_invoke = Groq(groq_api_key,selected_model)

        # response = obj_llm_invoke.invoke_llm_with_history(retriever,query)

        # return response

        
        
    elif selected_db == "Chroma":
        
        obj_chroma = ChromaVDB(main_config)
        retriver = obj_chroma.chroma_vectorstores_and_retriever(docs=docs)
        st.session_state["retriever"] = retriver
        
        
    else:
        st.error("Invalid database option selected.")
        result = None
    
    
    st.sidebar.success("Documents processed sucessfilly !!")


# Process PDF files
def process_pdf_files(main_config,selected_db,repo_path,groq_api_key,selected_model, rag_path_ext,query):
    if selected_db == "Qdrant":
        # LOAD
        obj_loaders = PDFLoader(repo_path, rag_path_ext)
        documents = obj_loaders.load_and_process_pdfs()

        # TRANSFORM
        obj_text_splitters = RecursiveSplitters()
        docs = obj_text_splitters.text_splitters(documents)

        #EMBEDDINGS and stored into VDB
        obj_qdr = QdrantVDB(main_config,repo_path,groq_api_key,selected_model, rag_path_ext,query)

        retriever = obj_qdr.vectorstores_and_get_context_retriever(docs)
        
        st.session_state["retriever"] = retriever

        # obj_llm_invoke = Groq(groq_api_key,selected_model)

        # response = obj_llm_invoke.invoke_llm_with_history(retriever,query)

        return response

        
        
    elif selected_db == "Chroma":
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
    
    st.session_state["retriever"] = retriever

    # obj_llm_invoke = Groq(groq_api_key,selected_model)

    # response = obj_llm_invoke.invoke_llm_with_history(retriever,query)
    return response


def load_streamlit_ui() : 
    # MAIN SCREEN
    st.set_page_config(page_title="🦜LangChain - Q&A with RAG ", layout="wide")
    st.header("🦜LangChain - Q&A with RAG ")
    

    # SIDEBAR
    
    user_input = {}
    selected_model = ""
    groq_api_key = ""

    #st.sidebar.header("Groq API Key")
    groq_api_key_input = st.sidebar.text_input("Enter Groq API Key", type="password")
    if groq_api_key_input:
        groq_api_key = groq_api_key_input  # Update placeholder with user-provided key
        st.sidebar.header("Model Selection (if applicable)")
        model_options = ["mixtral-8x7b-32768","llama3-8b-8192", "llama3-70b-8192"]  # Replace with your actual models
        selected_model = st.sidebar.selectbox("Select Model:", model_options)

    #st.sidebar.header("RAG Document Format")
    file_format = st.sidebar.selectbox("Select Input Document Format:", ["Code Repo", "PDF", "CSV"])

    # Check if the section exists
    if not main_config.has_section('Supported_Extensions'):
        raise ValueError("Section 'Supported_Extensions' not found in config.ini")

    # Extract extensions from the config dictionary
    extensions =  main_config['Supported_Extensions']['extensions']
    supported_extensions = eval(extensions)
    accepted_types = [f"{ext}" for ext in supported_extensions]

    # Print the list of extensions
    print(accepted_types)

    if file_format == "Code Repo":
        # st.sidebar.header("Select Language")
        # lang_options = ["java", "python", "csharp"]  # Replace with your actual models
        # selected_lang = st.sidebar.selectbox("Select Model:", lang_options)
        # if selected_lang == "java":
        #     rag_path_ext = ".java"
        # elif selected_lang == "python":
        #     rag_path_ext = ".py"
        # elif selected_lang == "csharp":
        #     rag_path_ext = ".cs"
        rag_path_ext =  accepted_types

    elif file_format == "PDF":
        rag_path_ext = ".pdf"

    elif file_format == "CSV":
        rag_path_ext = ".csv"



    uploaded_files = st.sidebar.file_uploader(f"Upload files", type=rag_path_ext,accept_multiple_files=True,)

    #st.sidebar.header("Vector Stores Options")
    selected_db = st.sidebar.selectbox("Select Vector DB:", ["Qdrant", "Chroma"])

    process_files = st.sidebar.button("Process")
    
    user_input = {"process_files":process_files,"file_format":file_format,"uploaded_files":uploaded_files,"groq_api_key":groq_api_key,"selected_model":selected_model,"selected_db":selected_db}
    
    return user_input


def user_query():
    # ENTER QUERY
    query = st.chat_input("Enter your query here")
    return query
    
def process_files_for_vectorization(user_input,query):
    # CHAT START
    process_files = user_input["process_files"]
    file_format = user_input["file_format"]
    uploaded_files = user_input["uploaded_files"]
    selected_model = user_input["uploaded_files"]
    selected_db = user_input["selected_db"]
    
    
    #if query :
    if process_files is None :
        st.warning("Please Process the File ")
        
    else :
        #st.chat_message("user").write(f"🧑‍💻: {query}")
        try : 
            result = ""
            if file_format == "Code Repo":
                if len(uploaded_files) > 0 :
                    repo_path = uploaded_files
                
                result = process_code_repo(main_config,selected_db,repo_path,groq_api_key,selected_model, rag_path_ext,query)

            elif file_format == "PDF":
                if len(uploaded_files) > 0 :
                    repo_path = uploaded_files
                result = process_pdf_files(main_config,selected_db,repo_path,groq_api_key,selected_model, rag_path_ext,query)
            elif file_format == "CSV":
                result = process_csv_files(uploaded_files,query)
            else:
                st.error("Invalid format selected.")

          

        except Exception as e:
            st.warning(f"Internal Server Error : {e}")
                
                
def has_empty_values(user_input):
  """
  Checks if any value in the dictionary is empty or "".

  Args:
      user_input (dict): The dictionary to check.

  Returns:
      bool: True if any value is empty or "", False otherwise.
  """
  # Use list comprehension to create a list of boolean values indicating emptiness
  empty_checks = [value == "" for value in user_input.values()]
  # Use any() to check if any element in the list is True (empty value)
  return any(empty_checks)


def initiate_chat_with_LLM(query,groq_api_key):
    obj_llm_invoke = Groq(groq_api_key,selected_model)
    retriever = st.session_state["retriever"]
    if retriever == "":
        response = obj_llm_invoke.llm_invoke_without_rag(query)
    else :
        response = obj_llm_invoke.invoke_llm_with_history(retriever,query)
    return response

    

# START EXECUTION
if __name__ == "__main__":
    user_input = load_streamlit_ui()
    st.chat_message("assistant").write("How can I help you Today?")
    query = user_query()
    # If not, then initialize it
    if 'retriever' not in st.session_state:
        st.session_state['retriever'] = ""
    
    
    
    # # Check for empty values
    # if has_empty_values(user_input) and user_input["process_files"]==False:
    #     #st.sidebar.Warning("Please fill in all required fields to Start RAG")
    #     print("empty")
    
    if has_empty_values(user_input) is not None and user_input["process_files"]==True:
        processed = process_files_for_vectorization(user_input,query)
    
    
    if  query :
        st.chat_message("human").write(f"{query}")
        response = ""
        response = initiate_chat_with_LLM(user_input["selected_db"],user_input["groq_api_key"],)
        if response is not None:
            st.chat_message("assistant").write(f"{response}")
            
        else:
            st.warning("Response is Empty")
            
    
    
    

    st.sidebar.info("**Note:**")
    st.sidebar.write(
        "- Replace placeholder groq API KEY by creating account https://console.groq.com/keys and select the model - default is mixtral"
        "- Chroma Db storage in local and can be view the chunk storage "
        "- Qdrant is used as In-Memory storage"
        "- Code repo file - https://github.com/Priyansh42/Lung-Cancer-Detection/blob/main/lcd_cnn.py"
    )



