from langchain_core.prompts import \
    (ChatPromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     MessagesPlaceholder)
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create a list of messages
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an AI chatbot having a conversation with a human on {topic}."
        ), # System level prompt for all conversations
        MessagesPlaceholder(
            variable_name="history"
        ), # Give information of chat history in memory
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ), # Insert the user input
    ]
)

openai_api_key = "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"
chain = prompt | ChatOpenAI(openai_api_key=openai_api_key,
                            max_tokens=20)


# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")

# Give initial ai message from Chatbot
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="human_input",
    history_messages_key="history",
)
config = {"configurable": {"session_id": "any"}}

# Get user input
# Using Walrus operator - 1. Assign chat_input to prompt and write if it is non-empty

# Render chat input and process it
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Assuming 'chain_with_history' and 'config' are defined
    response = chain_with_history.invoke(
        {"topic": "car", "human_input": prompt},
        config)
    st.chat_message("ai").write(response.content)
