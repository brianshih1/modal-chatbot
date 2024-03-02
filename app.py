import re
import os

def main():
    import streamlit as st

    st.title("Modal Chatbot")

    if st.button('Reindex Code'):
        print("CLICKED")
        
    docs = load_docs()
    vector_store = index_documents(docs)
    answer = chat(vector_store)

    st.text(answer)

def python_filter(path: str):
    rule = re.compile(r"^\./code/modal/cli/.*\.py$")
    return bool(rule.match(path))

def load_docs():
    from langchain_community.document_loaders import GitLoader
    from langchain.text_splitter import PythonCodeTextSplitter
    
    import streamlit as st

    text = st.text("Loading docs...")

    print("LOADING DOCS")
    repo_path = "./code"
    py_loader = None
    if os.path.exists(repo_path):
        py_loader = GitLoader(repo_path="./code", file_filter=python_filter)
    else: 
        py_loader = GitLoader(
            repo_path="./code", 
            clone_url="https://github.com/modal-labs/modal-client.git", 
            file_filter=python_filter)
    py_docs = py_loader.load()
    st.text(f"Docs Count: {len(py_docs)}")
    paths = list(map(lambda x: x.metadata["file_path"], py_docs))
    print(f"Paths: {paths}")

    split_py_docs = PythonCodeTextSplitter().split_documents(py_docs)
    
    print(f"Docs Count: {len(py_docs)}")

    print(f"Split Docs Count: {len(split_py_docs)}")
    print("Hello")
    return split_py_docs

def index_documents(docs):
    from langchain.vectorstores.faiss import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from pinecone import Pinecone, PodSpec

    

    os.environ["PINECONE"]
    print("Indexing documents")
    embeddings = HuggingFaceEmbeddings()
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Created vector store")
    vectorstore.add_documents(docs)
    print("Added documentsss")
    return vectorstore

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
    return result["answer"]

if __name__ == "__main__":
    main()
