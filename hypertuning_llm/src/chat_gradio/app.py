import sys
from pathlib import Path
import os
import re
from collections import deque

import gradio as gr
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared prompt.py in src/
from prompt import KNOWLEDGE_BASE, SYSTEM_PROMPT
from llm import Stats, stream_chat

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL")
# no silent retries: the error shows right away and the chat's retry button is the retry
client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="none", max_retries=0)

KB_LINES = [line[2:] for line in KNOWLEDGE_BASE.splitlines() if line.startswith("- ")]
STOPWORDS = {"the", "and", "for", "with", "what", "when", "where", "who", "apa", "yang", "jam", "ada",
             "di", "ke", "dan", "itu", "ini", "berapa", "siapa", "kapan", "mana"}
LOG = deque(maxlen=50)

HERO = """
<div id="hero">
  <span class="pill"><span class="dot"></span>Sat, 31 Oct 2026 · 08:30–17:00 WIB</span>
  <h1>AWS Community Day <span class="grad">Jakarta 2026</span></h1>
  <p>Ask about the agenda, speakers, venue or tickets. Answers come only from the event page.</p>
  <div class="chips">
    <span>📍 BINUS @Anggrek, Jakarta Barat</span>
    <span>👥 300+ attendees</span>
    <span>💬 Bahasa Indonesia &amp; English</span>
  </div>
</div>
"""

FOOTNOTE = """
<div id="footnote">
  Served by vLLM on EKS · <code>qwen-lora</code> can be wrong, check
  <a href="https://jakarta.awscommunity.id/comday/" target="_blank">jakarta.awscommunity.id/comday</a>.
  Chats are saved in this browser only.
</div>
"""

PLACEHOLDER = "### Halo! 👋\nTanya apa saja soal ComDay Jakarta 2026, atau pilih salah satu pertanyaan di bawah."

EXAMPLES = [
    "Talk soal LLM on EKS jam berapa dan di ruangan mana?",
    "Apa saja sesi setelah makan siang di Main Stage?",
    "Apakah tiketnya bisa di-refund?",
    "Siapa saja pembicara dari AWS?",
    "Is there free parking at the venue?",
]
EXAMPLE_LABELS = ["🕐 LLM on EKS talk", "🍱 After lunch, Main Stage", "🎟️ Ticket refund", "🎤 Speakers from AWS",
                  "🚗 Parking"]

# Gradio puts the "dark" class on body; the choice is kept in this browser's localStorage
APPLY_THEME_JS = """
() => {
  let theme = null;
  try { theme = localStorage.getItem("comday-theme"); } catch (e) {}
  if (theme) document.body.classList.toggle("dark", theme === "dark");
}
"""
TOGGLE_THEME_JS = """
() => {
  const dark = document.body.classList.toggle("dark");
  try { localStorage.setItem("comday-theme", dark ? "dark" : "light"); } catch (e) {}
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f6f6fa",
    body_background_fill_dark="#0b0d14",
    block_background_fill="#ffffff",
    block_background_fill_dark="#12141c",
    block_border_color="#e4e4ec",
    block_border_color_dark="#232838",
    input_background_fill="#ffffff",
    input_background_fill_dark="#161922",
    button_primary_background_fill="linear-gradient(90deg, #f97316, #ec4899, #a855f7)",
    button_primary_background_fill_dark="linear-gradient(90deg, #f97316, #ec4899, #a855f7)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ea580c, #db2777, #9333ea)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #ea580c, #db2777, #9333ea)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)


def retrieve(question: str, k: int = 5) -> list[str]:
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2 and w not in STOPWORDS}
    scored = [(sum(w in line.lower() for w in words), line) for line in KB_LINES]
    return [line for score, line in sorted(scored, key=lambda s: -s[0])[:k] if score]


def sources_message(lines: list[str], stats: Stats | None) -> dict:
    """Collapsible block above each answer, so the sources stay with it in the saved history."""
    title = f"📚 {len(lines)} sources" if lines else "📚 nothing matched on the event page"
    if stats:
        title += f" · first token {stats.ttft:.2f}s · {stats.tok_s:.0f} tok/s"
    body = "\n".join(f"- {line}" for line in lines) or "The answer should say it doesn't know."
    return {"role": "assistant", "content": body, "metadata": {"title": title, "status": "done"}}


def text_of(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def is_sources(message: dict) -> bool:
    return bool((message.get("metadata") or {}).get("title"))


def chat(message: str, history: list[dict]):
    lines = retrieve(message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": text_of(m["content"])} for m in history if not is_sources(m)]
    messages.append({"role": "user", "content": message})

    try:
        for answer, stats in stream_chat(client, messages):
            if stats:
                LOG.appendleft(stats.line())
            yield [sources_message(lines, stats), {"role": "assistant", "content": answer}], "\n".join(LOG)
    except APIConnectionError:
        raise gr.Error(f"Can't reach the model at {VLLM_URL}. Is the cluster up and VLLM_URL current?")
    except APIStatusError as e:
        raise gr.Error(f"The model returned HTTP {e.status_code}. A new GPU replica takes ~8 minutes to serve, "
                       "try again in a moment.")


with gr.Blocks(title="ComDay Jakarta 2026 Assistant") as demo:
    with gr.Row(elem_id="top"):
        gr.HTML(HERO)
        gr.Button("◐", elem_id="theme-toggle", scale=0, min_width=48).click(None, js=TOGGLE_THEME_JS)

    log = gr.Textbox(show_label=False, lines=8, max_lines=8, interactive=False, render=False, elem_id="log")
    gr.ChatInterface(
        chat,
        chatbot=gr.Chatbot(
            elem_id="chat",
            height="62vh",
            min_height=360,
            show_label=False,
            placeholder=PLACEHOLDER,
            buttons=["copy", "copy_all"],
        ),
        textbox=gr.Textbox(placeholder="Type a question…", show_label=False, submit_btn=True, stop_btn=True),
        examples=EXAMPLES,
        example_labels=EXAMPLE_LABELS,
        additional_outputs=[log],
        save_history=True,
        editable=True,
        flagging_mode="manual",
        fill_height=False,
    )
    with gr.Accordion("Live log · all users, newest first", open=False, elem_id="log-panel"):
        log.render()
    gr.HTML(FOOTNOTE)
    demo.load(None, js=APPLY_THEME_JS)

if __name__ == "__main__":
    demo.launch(theme=THEME, css_paths=[Path(__file__).with_name("style.css")])
