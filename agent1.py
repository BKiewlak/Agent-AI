from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, END, MessagesState 
from langgraph.prebuilt import ToolNode 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults 
from langchain_community.agent_toolkits.load_tools import load_tools
import os
import json

os.environ['TAVILY_API_KEY'] = "your_api-key"
os.environ['OPENAI_API_KEY'] = "your_api_key"


search_tool = TavilySearchResults(max_results=10, 
                                  include_answer=True, 
                                  include_raw_content=False)

human_tool = load_tools(["human"])[0]

tools = [search_tool, human_tool]


llm = ChatOpenAI(model="gpt-4o", max_tokens=None).bind_tools(tools)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def gather_data(state: MessagesState):

    messages = state["messages"]

    messages.append(SystemMessage(content='''
    Jesteś asystentem w pewnym banku.
    Klient zgłosił się z problemem. Twoim zadaniem jest zebranie danych o kliencie
    Musisz zebrać następujące dane:
    ** Imię i nazwisko
    ** poczta elektroniczna
    ** numer telefonu
    Do wyszukiwania informacji użyj narzędzia TavilySearchResults.
    Za pomocą HumanInputRun możesz dopytać klienta o jego dane oraz szczegóły dotyczące problemu.
      
    Odpowiedz w formacie JSON: 
    {{"case_data": wszystkie dane o kliencie zapisz i odpowiedz w formie tekstowej}}.
    W odpowiedzi nie mogą znajdować się żadne inne dane – 
    tylko poprawny, możliwy do przetworzenia JSON.
    '''))


    messages.append(response)
    state["messages"] = messages
    response = llm.invoke(messages)

    
    print(json.dumps(response.tool_calls, indent=2,ensure_ascii=False))
    print(json.dumps(response.content, indent=2,ensure_ascii=False))
   
    return {"messages": [response]}


tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)


workflow.add_node("gather_data_node", gather_data)
workflow.add_node("tools", tool_node)


workflow.set_entry_point("gather_data_node")

workflow.add_conditional_edges("gather_data_node", 
                               should_continue, 
                               ["tools", END])

workflow.add_edge("tools", "gather_data_node")

graph = workflow.compile()

prompt = """
    Oto opis sytuacji przedstawiony przez klienta:
    Chciałbym dowiedzieć się, jak otworzyć spółkę w Polsce?
    """

input_messages = [HumanMessage(prompt)]
output = graph.invoke({"messages": input_messages})


