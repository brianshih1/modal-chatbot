import modal
import time
import os
import re
import pathlib

from app import load_docs, index_documents, chat

stub = modal.Stub("git-repo", 
    image=modal.Image.debian_slim()
        .apt_install("git")
        .env({"GIT_PYTHON_REFRESH": "quiet"})
        .pip_install(
            "streamlit",
            "langchain",
            "langchain_openai",
            "GitPython",
            "faiss-cpu~=1.7.3",
            "sentence_transformers",
            "pinecone-client"
        )
        .pip_install("git+https://github.com/modal-labs/asgiproxy.git")
)

test_stub = modal.Stub("git-repo", 
    image=modal.Image.debian_slim()
        .apt_install("git")
        .env({"GIT_PYTHON_REFRESH": "quiet"})
        .pip_install(
            "langchain",
            "langchain_openai",
            "GitPython",
            "faiss-cpu~=1.7.3",
            "sentence_transformers",
            "pinecone-client"
        )
)

streamlit_script_local_path = pathlib.Path(__file__).parent / "app.py"
streamlit_script_remote_path = pathlib.Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)


HOST = "127.0.0.1"
PORT = "8000"


def spawn_server():
    import socket
    import subprocess

    process = subprocess.Popen(
        [
            "streamlit",
            "run",
            str(streamlit_script_remote_path),
            "--browser.serverAddress",
            HOST,
            "--server.port",
            PORT,
            "--browser.serverPort",
            PORT,
            "--server.enableCORS",
            "false",
        ]
    )

    # Poll until webserver accepts connections before running inputs.
    while True:
        try:
            socket.create_connection((HOST, int(PORT)), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(
                    f"launcher exited unexpectedly with code {retcode}"
                )

@stub.function(
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("pinecone-key")
    ]
)
@modal.asgi_app()
def run():
    from asgiproxy.config import BaseURLProxyConfigMixin, ProxyConfig
    from asgiproxy.context import ProxyContext
    from asgiproxy.simple_proxy import make_simple_proxy_app

    spawn_server()

    config = type(
        "Config",
        (BaseURLProxyConfigMixin, ProxyConfig),
        {
            "upstream_base_url": f"http://{HOST}:{PORT}",
            "rewrite_host_header": f"{HOST}:{PORT}",
        },
    )()
    proxy_context = ProxyContext(config)
    return make_simple_proxy_app(proxy_context)


@test_stub.function(
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount],
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("pinecone-key")
    ]
)
def index():
    docs = load_docs()
    vector_store = index_documents(docs)
    chat(vector_store)