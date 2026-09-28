from typing import Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt,Command
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import TypedDict
import requests

load_dotenv()
llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0)

#tools
@tool
def get_stocks_price(symbol: str)->dict:
    """
    Fetch latest stock price for a given symbol (eg: 'AAPL','TSLA')
    using Alpha Vantage with API key in the URL.
    """

    url="https://www.alphavantage.co/support/#api-key=V5RUYL0Q9G9EZBC4"
    r=requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol:str,quantity:int)->dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase , this tool will interrupt
    and wait for a human decision ("yes" /anything else).
    """

    decision=interrupt(f"approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision,str) and decision.lower() == "yes":
        return{
            "status":"success",
            "message":f"purchase order placed for {quantity} shares of {symbol}.",
            "symbol":symbol,
            "quantity":quantity,
        }

    else:
        return{
         "status":"cancelled",
         "message":f"purchase of {quantity} shares of {symbol} was declined by human.",
         "symbol":symbol,
         "quantity":quantity,   
        }

tools=[get_stocks_price,purchase_stock]
llm_with_tools=llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    """LLM node that may answer or request a tool call."""
    messages=state["messages"]
    response=llm_with_tools.invoke(messages)
    return {"messages":[response]}

tool_node=ToolNode(tools)

memory=MemorySaver()

graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")
chatbot=graph.compile(checkpointer=memory)

if __name__=='__main__':
    thread_id="demo_thread"

    while True:
        user_input=input("you: ")
        if user_input.lower().strip() in {"exit","quit"}:
            print("Goodbye!")
            break

        state={"messages":[HumanMessage(content=user_input)]}

        result=chatbot.invoke(
            state,
            config={"configurable":{"thread_id":"1234"}}
        )

        interrupts=result.get("__interrupt__",[])

        if interrupts:
            prompt_to_human= interrupts[0].value
            print(f"HITL : {prompt_to_human}")
            decision=input("your dicision:").strip().lower()

            result=chatbot.invoke(
                Command(resume=decision),
                config={"configurable":{"thread_id":thread_id}},
            )

        messages=result["messages"]
        last_msg=messages[-1]
        print(f"Bot: {last_msg.content}\n")
