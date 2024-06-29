#import 
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

from langchain_text_splitters import RecursiveCharacterTextSplitter

class GenericLoaders:
    def __init__(self,main_config,repo_path, rag_path_ext) -> None:
        self.main_config = main_config
        self.repo_path = repo_path
        self.rag_path_ext= rag_path_ext

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

        