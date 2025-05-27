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
    You are an assistant at a bank.
    A customer has reported a problem. Your task is to collect customer data.
    You must collect the following data:
    ** Full name
    ** Email address
    ** Phone number

    Use the TavilySearchResults tool to search for information.
    Use HumanInputRun to ask the customer for their data and details about the problem.
      
    Respond in the following JSON format:
    {{"case_data": enter all customer data and respond in text form here}}.
    The response must not contain any other data –
    only valid, processable JSON.
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
    Here is the situation description provided by the customer:
    
    """

input_messages = [HumanMessage(prompt)]
output = graph.invoke({"messages": input_messages})


