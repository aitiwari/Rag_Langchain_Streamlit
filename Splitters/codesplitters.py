#import 
from langchain_text_splitters import Language

from langchain_text_splitters import RecursiveCharacterTextSplitter

class CodeSplitters:
    def __init__(self,main_config,repo_path, rag_path_ext) -> None:
        self.main_config = main_config
        self.repo_path = repo_path
        self.rag_path_ext= rag_path_ext


    def code_splitters(self,documents):
        langlist = dict(self.main_config['Langlist'])

        try:
            selected_lang = "java" #default
            ext = self.rag_path_ext
            for language, ext in langlist.items():
                if ext.lower() == self.rag_path_ext.lower():
                    selected_lang= language

            if len(documents)>0:

                print(selected_lang)
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=selected_lang, chunk_size=2000, chunk_overlap=200
                )
                docs = splitter.split_documents(documents)
                print(len(docs))
                return docs

            else :
                raise ValueError("The data is empty.")
            
        except ValueError as e:
            print(e)
        










