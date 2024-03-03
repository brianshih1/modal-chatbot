import re
import os
import pickle
from modal import Volume

EMBEDDING_VOLUME = Volume.persisted("embeddings")
EMBEDDING_DIR = "/data"
EMBEDDING_FILE = "embedding.txt"

def python_filter(path: str):
    rule = re.compile(r"^\./code/modal/cli/.*\.py$")
    return bool(rule.match(path))

def load_docs():
    from langchain_community.document_loaders import GitLoader
    from langchain.text_splitter import PythonCodeTextSplitter
    
    print("LOADING DOCS")
    repo_path = "./code"
    py_loader = None
    # TODO: Use Volume
    if os.path.exists(repo_path):
        py_loader = GitLoader(repo_path="./code", file_filter=python_filter)
    else: 
        py_loader = GitLoader(
            repo_path="./code", 
            clone_url="https://github.com/modal-labs/modal-client.git", 
            file_filter=python_filter)
    py_docs = py_loader.load()
    paths = list(map(lambda x: x.metadata["file_path"], py_docs))
    print(f"Paths: {paths}")

    split_py_docs = PythonCodeTextSplitter().split_documents(py_docs)
    
    print(f"Docs Count: {len(py_docs)}")

    print(f"Split Docs Count: {len(split_py_docs)}")
    print("Hello")
    split_py_docs[0].metadata["file_name"]
    return split_py_docs

def index_documents():
    from langchain.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores.faiss import FAISS

    doc_embedding_path = f'{EMBEDDING_DIR}/{EMBEDDING_FILE}'

    embeddings = HuggingFaceEmbeddings()
    
    doc_embeddings = None
    if os.path.isfile(doc_embedding_path):
        print("Embeddings file found")
        with open(doc_embedding_path, 'rb') as f:
            doc_embeddings = pickle.load(f)
    else:
        docs = load_docs()
        print("Embeddings file not found")
        contents = list(map(lambda x: x.page_content, docs))
        filenames = list(map(lambda x: x.metadata["file_name"], docs))
        page_content_embeddings = embeddings.embed_documents(contents)
        doc_embeddings = list(zip(filenames, page_content_embeddings))
        with open(doc_embedding_path, 'wb') as f:
            pickle.dump(doc_embeddings, f)
        EMBEDDING_VOLUME.commit()
    print(f"Embeddings: {doc_embeddings[0]}")
    
    vectorstore = FAISS.from_embeddings(
        text_embeddings=doc_embeddings,
        embedding=embeddings
        )
    return vectorstore

def chat(vectorstore):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain_community.embeddings import HuggingFaceEmbeddings
   

    print("Start chat")
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
    )
    result = chain({"question": "What is Modal stub?", "chat_history": []})
    print(f"Result: {result}")
    return result["answer"]

