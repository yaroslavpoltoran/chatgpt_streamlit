import warnings

import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import config as cfg

warnings.filterwarnings("ignore")

load_dotenv()


def get_promt(input, history):
    promt = f"""You are a very useful AI assistant for an English teacher.

    Chat history: \n{history},
    User input: {input}
    assistant:"""
    return promt


llm = ChatOpenAI(model=cfg.MODEL)

st.set_page_config(page_title="ChatGPT Clone", page_icon="🤖", layout="wide")

st.title("ChatGPT Clone")

# Initialize session state for messages if it does not exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello there, I am a ChatGPT clone."}]

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input("Type your message here:")

if user_prompt:
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display user message
    with st.chat_message("user"):
        st.write(user_prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Concatenate all previous messages to form the chat history
            chat_history = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages][1:]
            chat_history = chat_history[-cfg.N_MESSAGES - 1 : -1]
            chat_history = "\n".join(chat_history)
            promt = get_promt(user_prompt, chat_history)
            print(promt)
            ai_response = llm.predict(promt).strip()
            st.write(ai_response)
            # Append AI response to session state
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
