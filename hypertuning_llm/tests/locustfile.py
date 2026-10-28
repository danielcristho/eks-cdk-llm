"""Locust load test against vLLM, with a live chart in the web UI (`just locust`).

Token counts show up as an extra "TOKENS" row: its "Average size" column is the average
completion tokens per request (Locust has no custom metric column, so response_length carries it).
"""
import random

from locust import HttpUser, constant, events, task

from load_test import QUESTIONS


class ChatUser(HttpUser):
    wait_time = constant(0)  # each user sends its next prompt as soon as the last answer lands

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
            events.request.fire(
                request_type="TOKENS", name="completion", response_time=0,
                response_length=usage["completion_tokens"], exception=None, context={},
            )
