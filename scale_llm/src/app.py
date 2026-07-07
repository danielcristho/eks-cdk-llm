import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")
MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="none")

st.set_page_config(page_title="Llama 3 Chatbot", page_icon="🦙")
st.title("🦙 Llama 3 Chatbot")
st.caption("Powered by vLLM on Ray Serve, autoscaled via Karpenter")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("How is you day? Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=st.session_state.messages,
            stream=True,
        )
        response = st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
