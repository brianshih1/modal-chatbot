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
    rule = re.compile(r"^\./code/modal/cli/.*\.py$")
    return bool(rule.match(path))

def txt_filter(path: str):
    rule = re.compile(r".+\.txt")
    return bool(rule.match(path))

@stub.function()
def load_docs(path):
    from langchain_community.document_loaders import GitLoader
    from langchain.text_splitter import PythonCodeTextSplitter
    py_loader = GitLoader(repo_path="./code", clone_url="https://github.com/modal-labs/modal-client.git", file_filter=python_filter)
    # py_loader = GitLoader(repo_path="./code", clone_url="https://github.com/gitpython-developers/QuickStartTutorialFiles.git", file_filter=python_filter)
    py_docs = py_loader.load()
    
    paths = list(map(lambda x: x.metadata["file_path"], py_docs))
    print(f"Paths: {paths}")

    split_py_docs = PythonCodeTextSplitter().split_documents(py_docs)
    
    print(f"Docs Count: {len(py_docs)}")

    print(f"Split Docs Count: {len(split_py_docs)}")
    print("Hello")
    return split_py_docs
    

# TODO: Can we use volume to not have to reindex every time?
@stub.function(secrets=[modal.Secret.from_name("openai-secret")])
def index_documents(docs):
    from langchain.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("Indexing documents")
    embeddings = HuggingFaceEmbeddings()
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Created vector store")
    vectorstore.add_documents(docs)
    print("Added documentsss")
    chat.local(vectorstore)
    return vectorstore
    
@stub.function(secrets=[modal.Secret.from_name("openai-secret")])
def chat(vectorstore):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain

    print("Start chat")
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
    )

    result = chain({"question": "What is Modal stub?", "chat_history": []})

    print(f"Result: {result}")

@stub.local_entrypoint()
def main():
    docs = load_docs.remote("/code")
    vectorstore = index_documents.remote(docs)
    # chat.remote(vectorstore)
    time.sleep(10)