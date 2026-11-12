
import streamlit as st

from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="LangGraph Ollama Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")
st.caption("Powered by Ollama + LangGraph")

# Conversation configuration
CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}

# Initialize chat history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:

    # Display user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = chatbot.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG
            )

            ai_message = response["messages"][-1].content

            st.write(ai_message)

    # Save assistant response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
