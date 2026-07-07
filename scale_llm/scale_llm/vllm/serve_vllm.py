"""
serve_vllm.py — Ray Serve entrypoint for vLLM.

Mounted into each Ray worker at /serve-scripts/serve_vllm.py via ConfigMap.
Ray Serve loads it via: import_path: serve_vllm:build_app
"""

import asyncio
from typing import Optional

# Runtime imports — only available inside the Ray worker container.
# try/except lets this file be imported at CDK synth time for ConfigMap embedding.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from ray import serve
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    _RUNTIME_IMPORTS_OK = True
except ImportError:
    _RUNTIME_IMPORTS_OK = False

if _RUNTIME_IMPORTS_OK:
    app = FastAPI(title="vLLM on Ray Serve", version="1.0.0")

    @serve.deployment(
        ray_actor_options={"num_gpus": 1, "memory": 12 * 1024 ** 3},
        autoscaling_config={
            "min_replicas": 0,
            "max_replicas": 4,
            "target_ongoing_requests": 2,
        },
        max_ongoing_requests=5,
    )
    @serve.ingress(app)
    class VLLMDeployment:

        def __init__(
            self,
            model: str,
            dtype: str = "half",
            quantization: str = "awq",
            max_model_len: int = 4096,
            download_dir: str = "/model-cache",
            gpu_memory_utilization: float = 0.90,
        ):
            engine_args = AsyncEngineArgs(
                model=model,
                dtype=dtype,
                quantization=quantization,
                max_model_len=max_model_len,
                download_dir=download_dir,
                gpu_memory_utilization=gpu_memory_utilization,
                # Disable vLLM's internal Ray to avoid conflict with KubeRay
                worker_use_ray=False,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._serving_chat: Optional[OpenAIServingChat] = None

        async def _get_serving_chat(self) -> OpenAIServingChat:
            if self._serving_chat is None:
                model_config = await self.engine.get_model_config()
                self._serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    served_model_names=[model_config.model],
                    response_role="assistant",
                )
            return self._serving_chat

        @app.get("/health")
        async def health(self) -> dict:
            return {"status": "ok"}

        @app.get("/v1/models")
        async def list_models(self) -> dict:
            model_config = await self.engine.get_model_config()
            return {
                "object": "list",
                "data": [{"id": model_config.model, "object": "model", "owned_by": "vllm"}],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(self, request: ChatCompletionRequest, raw_request: Request):
            serving_chat = await self._get_serving_chat()
            generator = await serving_chat.create_chat_completion(request, raw_request)
            if request.stream:
                return StreamingResponse(generator, media_type="text/event-stream")
            return generator


def build_app(
    model: str,
    dtype: str = "half",
    quantization: str = "awq",
    max_model_len: int = 4096,
    download_dir: str = "/model-cache",
    gpu_memory_utilization: float = 0.90,
):
    """Entry point called by Ray Serve — args map to the serveConfigV2 args block."""
    return VLLMDeployment.bind(
        model=model,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        download_dir=download_dir,
        gpu_memory_utilization=gpu_memory_utilization,
    )
