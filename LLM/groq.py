from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI

# llm = ChatGroq(groq_api_key ="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" )
# print(llm.invoke("hi"))

class Groq:
    def __init__(self,groq_api_key,selected_model) -> None:
        self.groq_api_key = groq_api_key
        self.selected_model = selected_model

    def get_llm(self):
        llm = ChatGroq(groq_api_key =self.groq_api_key,model=self.selected_model )
        return llm

    def invoke_llm_with_history(self,retriever,query):

        llm = self.get_llm()
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                ),
            ]
        )

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                ("placeholder", "{chat_history}"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, prompt)

        qa = create_retrieval_chain(retriever_chain, document_chain)

        result = qa.invoke({"input": query})
        print(result)
        return result["answer"]



