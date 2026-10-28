import sys
from pathlib import Path
import os
import re
from collections import deque

import gradio as gr
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared prompt.py in src/
from prompt import KNOWLEDGE_BASE, SYSTEM_PROMPT
from llm import stream_chat

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")
client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="none")

KB_LINES = [line[2:] for line in KNOWLEDGE_BASE.splitlines() if line.startswith("- ")]
STOPWORDS = {"the", "and", "for", "with", "what", "when", "where", "who", "apa", "yang", "jam", "ada",
             "di", "ke", "dan", "itu", "ini", "berapa", "siapa", "kapan", "mana"}
LOG = deque(maxlen=50)

HERO = """
<div id="hero">
  <h1><mark>ComDay Jakarta 2026</mark> assistant</h1>
  <p>Sat, 31 Oct 2026 · BINUS @Anggrek, Jakarta Barat · answers come only from the event page</p>
</div>
"""

EXAMPLES = [
    "Talk soal LLM on EKS jam berapa dan di ruangan mana?",
    "Apa saja sesi setelah makan siang di Main Stage?",
    "Apakah tiketnya bisa di-refund?",
    "Siapa saja pembicara dari AWS?",
    "Is there free parking at the venue?",
]


def retrieve(question: str, k: int = 5) -> list[str]:
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2 and w not in STOPWORDS}
    scored = [(sum(w in line.lower() for w in words), line) for line in KB_LINES]
    return [line for score, line in sorted(scored, key=lambda s: -s[0])[:k] if score]


def format_sources(lines: list[str]) -> str:
    if not lines:
        return "**sources**\n\nnothing matched, so the answer should say it doesn't know."
    return "**sources**\n\n" + "\n".join(f"- {line}" for line in lines) + \
        "\n\n[jakarta.awscommunity.id/comday](https://jakarta.awscommunity.id/comday/)"


def text_of(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def chat(message: str, history: list[dict]):
    sources = format_sources(retrieve(message))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": text_of(m["content"])} for m in history]
    messages.append({"role": "user", "content": message})

    try:
        for answer, stats in stream_chat(client, messages):
            if stats:
                LOG.appendleft(stats.line())
            yield answer, sources, "\n".join(LOG)
    except APIConnectionError:
        raise gr.Error(f"Can't reach the model at {VLLM_URL}. Is the cluster up and VLLM_URL current?")


with gr.Blocks(title="ComDay Jakarta 2026 Assistant") as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            sources = gr.Markdown(
                "**sources**\n\nthe event page entries behind each answer show up here.",
                render=False,
                elem_id="sources",
            )
            log = gr.Textbox(
                label="live log (all users, newest first)",
                lines=10,
                max_lines=10,
                interactive=False,
                render=False,
                elem_id="log",
            )
            gr.ChatInterface(
                chat,
                chatbot=gr.Chatbot(
                    elem_id="chat",
                    height=500,
                    show_label=False,
                    placeholder="ask about the agenda, speakers, venue or tickets",
                ),
                textbox=gr.Textbox(placeholder="type a question...", scale=7),
                examples=EXAMPLES,
                additional_outputs=[sources, log],
                fill_height=False,
            )
        with gr.Column(scale=2):
            sources.render()
            log.render()

if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Patrick Hand"), "Comic Sans MS", "sans-serif"],
    )
    demo.launch(theme=theme, css_paths=[Path(__file__).with_name("style.css")])
