import modal
import time
import os
import re

stub = modal.Stub("git-repo", 
    image=modal.Image.debian_slim()
        .apt_install("git")
        .env({"GIT_PYTHON_REFRESH": "quiet"})
        .pip_install(
            "langchain",
            "langchain_openai",
            "GitPython",
            "faiss-cpu~=1.7.3",
            "sentence_transformers"
        )

)

def file_filter(path: str):
    # TODO: regex
    return path.endswith(".md") or path.endswith(".py") or path.endswith(".txt")

def markdown_filter(path: str):
    rule = re.compile(r".+\.md")
    return bool(rule.match(path))

def python_filter(path: str):
    rule = re.compile(r".+\.py")
    return bool(rule.match(path))

def txt_filter(path: str):
    rule = re.compile(r".+\.txt")
    return bool(rule.match(path))

@stub.function()
def load_files(path):
    from langchain_community.document_loaders import GitLoader
    from langchain.text_splitter import PythonCodeTextSplitter
    py_loader = GitLoader(repo_path="./code", clone_url="https://github.com/modal-labs/modal-client.git", file_filter=python_filter)
    py_docs = py_loader.load()
    
    split_py_docs = PythonCodeTextSplitter().split_documents(py_docs)
    print(f"Docs Count: {len(py_docs)}")
    print(f"Split Docs Count: {len(split_py_docs)}")
    print("Hello")
    index_documents.remote(split_py_docs)
    

# TODO: Can we use volume to not have to reindex every time?
@stub.function(secrets=[modal.Secret.from_name("my-openai-secret")])
def index_documents(docs):
    from langchain.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings

    # embeddings = OpenAIEmbeddings(api_key=os.environ["open_api_key"])
    embeddings = HuggingFaceEmbeddings()
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.add_documents(docs)
    # vectorstore.
    # print(f"Results: {results}")


@stub.local_entrypoint()
def main():
    load_files.remote("/code")
    print("foo")
    time.sleep(599)