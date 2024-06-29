from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
class CSVAgent:
    def __init__(self,llm) -> None:
        self.llm = llm
     
    def create_agent(self,csv_file):
        try : 
            agent = create_csv_agent(
                    self.llm, csv_file , verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,)
        except Exception as e:
            print(e)
        return agent
    
    def run_agent(self,agent,query):
        result = agent.run(query)
        return result