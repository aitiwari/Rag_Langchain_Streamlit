import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DataFrameLoader

class CSVLoaders:
    def __init__(self,file_path):
        self.file_path = file_path


    def csv_loader(self):
        try :

            uploaded_file = self.file_path
            documents = []
            dfs = []
            
            if uploaded_file :
                import pandas as pd
                import streamlit as st
                for file in uploaded_file:
                    dataframe = pd.read_csv(file)
                   # st.write(dataframe)
                    dfs.extend(dataframe)
                    #source column can be selected and pass as page_content by user, defalut - column 0 
                    loader = DataFrameLoader(dataframe, page_content_column=dataframe.columns[0])
                    documents.extend(loader.load())
            else :
                raise ValueError("Error Occured in loading CSV files")
        
        
        except Exception as e:
            print(e)
        return documents


        