import re
import os
import pickle
from modal import Volume
from typing import Callable

EMBEDDING_VOLUME = Volume.persisted("embeddings")
EMBEDDING_DIR = "/data"
EMBEDDING_FILE = "embedding.txt"

def modal_client_filter(path: str):
    rule = re.compile(r"^\./code/modal/cli/.*\.py$")
    return bool(rule.match(path))

def modal_examples_filter(path: str):
    rule = re.compile(r".*\.py$")
    return bool(rule.match(path))

def load_docs():
    modal_client_docs = load_docs_from_url("./modal-client", "https://github.com/modal-labs/modal-client.git", modal_client_filter)
    modal_examples_docs = load_docs_from_url("./modal-examples", "https://github.com/modal-labs/modal-examples.git", modal_examples_filter)
    return modal_client_docs + modal_examples_docs

def load_docs_from_url(repo_path: str, git_url: str, filter: Callable[[str], bool]):
    from langchain_community.document_loaders import GitLoader
    from langchain.text_splitter import PythonCodeTextSplitter
    
    py_loader = None
    if os.path.exists(repo_path):
        py_loader = GitLoader(repo_path=repo_path, file_filter=filter)
    else: 
        py_loader = GitLoader(
            repo_path=repo_path, 
            clone_url=git_url, 
            file_filter=filter)
    py_docs = py_loader.load()
    paths = list(map(lambda x: x.metadata["file_path"], py_docs))

    split_py_docs = PythonCodeTextSplitter().split_documents(py_docs)
    return split_py_docs

def index_documents(force_reindex: bool):
    from langchain.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores.faiss import FAISS

    doc_embedding_path = f'{EMBEDDING_DIR}/{EMBEDDING_FILE}'

    embeddings = HuggingFaceEmbeddings()
    
    doc_embeddings = None
    if os.path.isfile(doc_embedding_path) and not force_reindex:
        print("Not indexing code. Embeddings file found.")
        with open(doc_embedding_path, 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        if os.path.exists(doc_embedding_path):
            os.remove(doc_embedding_path)
        docs = load_docs()
        print("Indexing code")
        contents = list(map(lambda x: x.page_content, docs))
        page_content_embeddings = embeddings.embed_documents(contents)
        doc_embeddings = list(zip(contents, page_content_embeddings))
        with open(doc_embedding_path, 'wb') as f:
            pickle.dump(doc_embeddings, f)
        EMBEDDING_VOLUME.commit()
    
    vectorstore = FAISS.from_embeddings(
        text_embeddings=doc_embeddings,
        embedding=embeddings
        )
    
    return vectorstore

def chat(vectorstore, question: str, history: list[tuple[str, str]]):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain_community.embeddings import HuggingFaceEmbeddings
   
    chat_model = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        api_key=os.environ["OPENAI_API_KEY"]
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
    )

    return chain({
        "question": question, 
        "chat_history": history
        })


