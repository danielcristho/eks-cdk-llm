"""Gradio chat UI in the cluster, so chat traffic reaches vLLM through the Service that zonal shift steers."""
import hashlib
import pathlib
import re

from aws_cdk import Stack, CfnOutput, aws_eks as eks
from constructs import Construct

PROJECT_DIR = pathlib.Path(__file__).parents[2]
SRC_DIR = PROJECT_DIR / "src"
# ConfigMap key -> path under /app/src, the same layout as src/ so app.py finds prompt.py and llm.py
APP_FILES = {
    "app.py": "chat_gradio/app.py",
    "style.css": "chat_gradio/style.css",
    "llm.py": "llm.py",
    "prompt.py": "prompt.py",
    "knowledge_base.md": "knowledge_base.md",
}
PACKAGES = ("gradio", "openai", "python-dotenv")
IMAGE = "python:3.12-slim"
NAMESPACE = "kuberay"
VLLM_URL = "http://vllm-serve-svc.kuberay.svc.cluster.local:8000"


class ChatStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        cluster: eks.Cluster,
        kuberay_operator,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        files = {key: (SRC_DIR / path).read_text(encoding="utf-8") for key, path in APP_FILES.items()}
        # a change to any app file changes the pod template, so the pod restarts with the new ConfigMap
        files_sha = hashlib.sha256("".join(files.values()).encode()).hexdigest()[:12]
        pip_install = "pip install --no-cache-dir --root-user-action=ignore " + " ".join(
            f"'{pin}'" for pin in _requirement_pins(PACKAGES)
        )
        labels = {"app": "chat-ui"}

        chat = cluster.add_manifest(
            "ChatUi",
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "chat-ui", "namespace": NAMESPACE},
                "data": files,
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "chat-ui", "namespace": NAMESPACE},
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": labels},
                    "template": {
                        "metadata": {"labels": labels, "annotations": {"chat-ui/files-sha": files_sha}},
                        "spec": {
                            "nodeSelector": {"role": "system"},
                            "containers": [{
                                "name": "chat-ui",
                                "image": IMAGE,
                                # .gradio/ (like/dislike feedback) is written to the working directory
                                "workingDir": "/tmp",
                                "command": ["sh", "-c", f"{pip_install} && exec python /app/src/chat_gradio/app.py"],
                                "env": [
                                    {"name": "VLLM_URL", "value": VLLM_URL},
                                    {"name": "GRADIO_SERVER_NAME", "value": "0.0.0.0"},
                                    {"name": "GRADIO_SERVER_PORT", "value": "7860"},
                                    {"name": "GRADIO_ANALYTICS_ENABLED", "value": "False"},
                                ],
                                "ports": [{"name": "http", "containerPort": 7860}],
                                "resources": {
                                    "requests": {"cpu": "250m", "memory": "512Mi"},
                                    "limits": {"cpu": "1", "memory": "1Gi"},
                                },
                                # pip install runs before the server comes up, ~1-2 min on a fresh pod
                                "startupProbe": {
                                    "httpGet": {"path": "/", "port": "http"},
                                    "periodSeconds": 10,
                                    "failureThreshold": 30,
                                },
                                "readinessProbe": {"httpGet": {"path": "/", "port": "http"}, "periodSeconds": 10},
                                "volumeMounts": [{"name": "app", "mountPath": "/app/src"}],
                            }],
                            "volumes": [{
                                "name": "app",
                                "configMap": {
                                    "name": "chat-ui",
                                    "items": [{"key": key, "path": path} for key, path in APP_FILES.items()],
                                },
                            }],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "chat-ui", "namespace": NAMESPACE},
                "spec": {"selector": labels, "ports": [{"name": "http", "port": 80, "targetPort": "http"}]},
            },
        )
        chat.node.add_dependency(kuberay_operator)

        CfnOutput(
            self, "ChatUiNote",
            value="just chat-ui, then open http://localhost:7860",
            description="How to reach the Gradio chat UI (in-cluster, no public endpoint)",
        )


def _requirement_pins(names: tuple[str, ...]) -> list[str]:
    """Pins for the chat UI's packages, read from requirements.txt so versions live in one place."""
    pins = {}
    for line in (PROJECT_DIR / "requirements.txt").read_text().splitlines():
        name = re.split(r"[<>=!~\s]", line, maxsplit=1)[0]
        if name in names:
            pins[name] = line.strip()
    missing = set(names) - pins.keys()
    if missing:
        raise ValueError(f"requirements.txt has no pin for {sorted(missing)}")
    return list(pins.values())
