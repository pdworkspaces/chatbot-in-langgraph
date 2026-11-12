
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

# Initialize Ollama LLM
llm = ChatOllama(
    model="llama3",
    temperature=0.7,
)

# Define chatbot state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define chatbot node
def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": [response]}


# Initialize checkpointer
checkpointer = InMemorySaver()


# Build LangGraph workflow
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# Compile chatbot
chatbot = graph.compile(checkpointer=checkpointer)


# Optional backend test
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "test-thread"
        }
    }

    response = chatbot.invoke(
        {"messages": [{"role": "user", "content": "Hello, who are you?"}]}, # type: ignore
        config=config # type: ignore
    )

    print(response["messages"][-1].content)
