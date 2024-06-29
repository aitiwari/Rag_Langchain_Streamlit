#import 
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveSplitters:
    def __init__(self) -> None:
        pass


    def text_splitters(self,documents):

        try :
            if len(documents)>0:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                print(len(docs))
                return docs
            else :
                raise ValueError("The data is empty.")
            
        except Exception as e:
            print(e)










