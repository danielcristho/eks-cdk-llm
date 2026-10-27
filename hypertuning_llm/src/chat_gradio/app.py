import sys
from pathlib import Path
import os
import re

import gradio as gr
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared prompt.py in src/
from prompt import KNOWLEDGE_BASE, SYSTEM_PROMPT

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")
client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="none")

KB_LINES = [line[2:] for line in KNOWLEDGE_BASE.splitlines() if line.startswith("- ")]
STOPWORDS = {"the", "and", "for", "with", "what", "when", "where", "who", "apa", "yang", "jam", "ada",
             "di", "ke", "dan", "itu", "ini", "berapa", "siapa", "kapan", "mana"}

CSS = """
:root, .dark { --ink: #2b2b2b; --muted: #6b6b6b; --paper: #fbfaf6; --line: #a0a0a0; --mark: #ffe9a8; }
body, gradio-app, .gradio-container { background: var(--paper) !important; }
.gradio-container { max-width: 1100px !important; margin: auto; color: var(--ink); }
.gradio-container * { box-shadow: none !important; }
.block, .form, .panel { background: transparent !important; border: none !important; }
#hero { padding: 8px 4px 14px; border-bottom: 2px dashed var(--line); }
#hero h1 { margin: 0; font-size: 1.9rem; color: var(--ink); }
#hero h1 mark { background: var(--mark); padding: 0 6px; }
#hero p { margin: 4px 0 0; color: var(--muted); }
#chat { border: 2px solid var(--ink) !important; border-radius: 14px 18px 12px 16px !important; }
#sources { border: 2px dashed var(--line) !important; border-radius: 12px; padding: 4px 14px !important; }
#sources li { margin-bottom: 6px; }
textarea { border: 2px solid var(--ink) !important; border-radius: 12px !important; background: #fff !important; }
.message.bot { background: #fff !important; border: 1.5px solid var(--line) !important; }
.message.user { background: var(--mark) !important; border: 1.5px solid var(--ink) !important; }
button.example { border: 1.5px dashed var(--line) !important; background: #fff !important; }
footer { display: none !important; }
"""

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
    # Gradio 6 history content can be a list of parts
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def chat(message: str, history: list[dict]):
    sources = format_sources(retrieve(message))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": text_of(m["content"])} for m in history]
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(model="qwen-lora", messages=messages, stream=True)
        answer = ""
        for chunk in stream:
            answer += chunk.choices[0].delta.content or ""
            yield answer, sources
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
                additional_outputs=[sources],
                fill_height=False,
            )
        with gr.Column(scale=2):
            sources.render()

if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("Patrick Hand"), "Comic Sans MS", "sans-serif"],
    )
    demo.launch(theme=theme, css=CSS)
