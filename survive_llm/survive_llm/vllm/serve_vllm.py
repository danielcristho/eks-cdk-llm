"""
Ray Serve entrypoint for vLLM.
Mounted into each Ray worker at /serve-scripts/serve_vllm.py via ConfigMap.
"""

from typing import Optional

# Runtime imports, available inside the ray worker container.
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from ray import serve
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
    _RUNTIME_IMPORTS_OK = True
except ImportError:
    _RUNTIME_IMPORTS_OK = False

if _RUNTIME_IMPORTS_OK:
    app = FastAPI(title="vLLM on Ray Serve", version="1.0.0")

    # min_replicas=0: no GPU worker exists until a request actually needs one
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
            served_model_name: Optional[str] = None,
        ):
            engine_args = AsyncEngineArgs(
                model=model,
                served_model_name=served_model_name,
                dtype=dtype,
                quantization=quantization,
                max_model_len=max_model_len,
                download_dir=download_dir,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._serving_chat: Optional[OpenAIServingChat] = None

        async def _get_serving_chat(self) -> OpenAIServingChat:
            if self._serving_chat is None:
                model_config = await self.engine.get_model_config()
                models = OpenAIServingModels(
                    engine_client=self.engine,
                    model_config=model_config,
                    base_model_paths=[BaseModelPath(name=model_config.served_model_name, model_path=model_config.model)],
                )
                self._serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    models,
                    response_role="assistant",
                    request_logger=None,
                    chat_template=None,
                    chat_template_content_format="auto",
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
                "data": [{"id": model_config.served_model_name, "object": "model", "owned_by": "vllm"}],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(self, request: ChatCompletionRequest, raw_request: Request):
            serving_chat = await self._get_serving_chat()
            generator = await serving_chat.create_chat_completion(request, raw_request)
            if request.stream:
                return StreamingResponse(generator, media_type="text/event-stream")
            return generator


def build_app(args: dict):
    return VLLMDeployment.bind(
        model=args["model"],
        dtype=args.get("dtype", "half"),
        quantization=args.get("quantization", "awq"),
        max_model_len=args.get("max_model_len", 4096),
        download_dir=args.get("download_dir", "/model-cache"),
        gpu_memory_utilization=args.get("gpu_memory_utilization", 0.90),
        served_model_name=args.get("served_model_name"),
    )
