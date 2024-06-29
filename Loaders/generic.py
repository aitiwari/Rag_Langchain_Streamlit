#import 
import os
import tempfile
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

from langchain_text_splitters import RecursiveCharacterTextSplitter

class GenericLoaders:
    def __init__(self,main_config,repo_path, rag_path_ext) -> None:
        self.main_config = main_config
        self.repo_path = repo_path
        self.rag_path_ext= rag_path_ext
    
    #when files are present in dir
    def loaders(self):
        # Load
        try :

            print("********self.repo_path*******")
            print(self.repo_path)
            print("********self.rag_path_ext*******")
            print(self.rag_path_ext)

            loader = GenericLoader.from_filesystem(
                self.repo_path,
                glob="**/*",
                suffixes=[self.rag_path_ext],
                exclude=["**/non-utf8-encoding.py"],
                parser=LanguageParser(parser_threshold=500),
            )
            documents = loader.load()
            print(len(documents))
        except Exception as e:
            print(e)
            
        return documents

    
    # which files are uploading in streamlit
    def code_loader_and_splitters(self):
        documents = []
        docs=[]
        try : 
            code_files = self.repo_path 
            langlist = dict(self.main_config['Langlist'])
            langlist = eval(langlist["lang_exts"])  # Convert the string to a dictionary
            if code_files:
                for code_file in code_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(code_file.read())

                    lang_ext = str(code_file.name.split(".")[-1])
                    lang_ext = str('.'+lang_ext)
                    tmp_file_path = tmp_file.name+lang_ext
                    tmp_file_path=tmp_file_path.replace("\\","/")
                    os.rename(tmp_file.name,tmp_file_path )
                    loader = GenericLoader.from_filesystem(
                            tmp_file_path,
                            glob="**/*",
                            suffixes=[lang_ext],
                            exclude=["**/non-utf8-encoding.py"],
                            parser=LanguageParser(parser_threshold=500),
                        )
                    
                    
                    selected_lang =""
                    for lang, ext in langlist.items():
                        if ext.lower() == lang_ext:
                            selected_lang = lang
                            print(f"selected lang: {selected_lang}")  # For debugging

                    if selected_lang =="":
                        raise ValueError("Issue in evaluating techctack from extension")
                    documents.extend(loader.load())
                    
                    splitter = RecursiveCharacterTextSplitter.from_language(
                    language=selected_lang, chunk_size=2000, chunk_overlap=200
                    )
                    docs.extend(splitter.split_documents(documents))

                    # Clean up temporary file
                    os.remove(tmp_file_path)
            
        except Exception as e:
            raise ValueError(f"Problem Occured in loader and splitter : {e}")
            
        return docs


        