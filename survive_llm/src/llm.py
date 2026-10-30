import time
from dataclasses import dataclass


@dataclass
class Stats:
    ttft: float  # seconds until the first token
    total: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def tok_s(self) -> float:
        return self.completion_tokens / max(self.total - self.ttft, 1e-6)

    def line(self) -> str:
        return (f"{time.strftime('%H:%M:%S')}  ttft {self.ttft:.2f}s  total {self.total:.1f}s  "
                f"prompt {self.prompt_tokens}  completion {self.completion_tokens}  {self.tok_s:.0f} tok/s")


def stream_chat(client, messages: list[dict]):
    """Yield (answer_so_far, None) per token, then (answer, Stats) once vLLM reports usage."""
    start = time.time()
    ttft = None
    answer = ""
    stream = client.chat.completions.create(
        model="qwen-lora",
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta and ttft is None:
                ttft = time.time() - start
            answer += delta
            yield answer, None
        if chunk.usage:  # last chunk, no choices
            yield answer, Stats(ttft or 0.0, time.time() - start,
                                chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
