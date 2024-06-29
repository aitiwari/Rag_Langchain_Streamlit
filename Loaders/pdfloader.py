import tempfile
import streamlit as st
import os
from glob import iglob
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self,repo_path, rag_path_ext) -> None:
        self.repo_path = repo_path
        self.rag_path_ext= rag_path_ext
    

    # Cache the function to load and process PDF documents
    #@st.cache(allow_output_mutation=True)
    def load_and_process_pdfs(self):
        documents = []
        try : 
            pdf_files = self.repo_path 
            # for filepath in iglob(os.path.join(pdf_folder_path, "**/*.pdf"), recursive=True):
            #     filepath = filepath.replace("\\","/")
            #     pdf_files.append(filepath)
            #     # bewlo is for extract image
            #     #loader = PyPDFLoader(filepath,extract_images=True) 
            #     loader = PyPDFLoader(filepath)

            #     documents.extend(loader.load())
            if pdf_files:
                for pdf_file in pdf_files:
                    # Save the file temporarily
                    #pdf_file  = pdf_file.replace("\\","/")
                    # tmp_location = os.path.join('/tmp', pdf_file.name)
                    # tmp_location = tmp_location.replace("\\","/")
                    # with open(tmp_location, 'wb') as tmp_file:
                    #     tmp_file.write(pdf_file.read())
                    # Save the file temporarily
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(pdf_file.read())
                        

                    
                    # tmp_file_path=tmp_file_path.replace("\\","/")

                    # with open(tmp_file_path, "wb") as temp_file:
                    #       temp_file.write(pdf_file.read())

                    # Load PDF using PyPDFLoader
                    tmp_file_path = tmp_file.name+".pdf"
                    tmp_file_path=tmp_file_path.replace("\\","/")
                    os.rename(tmp_file.name,tmp_file_path )
                    loader = PyPDFLoader(tmp_file_path)

                    # Load PDF using PyPDFLoader
                    #pages = loader.load_and_split()
                    documents.extend(loader.load())

                    # Process each page (you can add your custom logic here)
                    # for page_num, page_content in enumerate(pages):
                    #     st.write(f"Page {page_num + 1} content:")
                    #     st.write(page_content)

                    # # Clean up temporary file
                    os.remove(tmp_file_path)
            
        except Exception as e:
            print(e)
        return documents
        