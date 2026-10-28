import sys
from pathlib import Path
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared prompt.py in src/
from prompt import SYSTEM_PROMPT
from llm import stream_chat

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")

client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="none")

st.set_page_config(page_title="Qwen LoRA Chatbot", page_icon="🤖")
st.title("🤖 Qwen LoRA Chatbot")
st.caption("Powered by vLLM was built on top of Ray cluster, autoscaled via Karpenter")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("stats"):
            st.caption(msg["stats"])

if prompt := st.chat_input("How is your day? Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response, line = "", None
        for response, stats in stream_chat(client, [{"role": "system", "content": SYSTEM_PROMPT}] + history):
            placeholder.write(response)
            if stats:
                line = stats.line()
                st.caption(line)

    st.session_state.messages.append({"role": "assistant", "content": response, "stats": line})
