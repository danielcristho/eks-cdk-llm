"""Locust load test against vLLM (`just locust`)."""
import random

from locust import HttpUser, constant, events, task

from load_test import QUESTIONS


class ChatUser(HttpUser):
    wait_time = constant(0)

    @task
    def chat(self):
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-lora",
                "messages": [{"role": "user", "content": random.choice(QUESTIONS)}],
                "max_tokens": 256,
            },
            name="chat",
            timeout=300,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            usage = resp.json()["usage"]
            # no custom metric column in Locust: the TOKENS row's "Average size" is completion tokens
            events.request.fire(
                request_type="TOKENS", name="completion", response_time=0,
                response_length=usage["completion_tokens"], exception=None, context={},
            )
