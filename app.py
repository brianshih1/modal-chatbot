from data import load_docs, index_documents, chat

def main():
    import streamlit as st

    st.title("Modal Chatbot")

    if st.button('Reindex Code'):
        print("CLICKED")
        
    docs = load_docs()
    vector_store = index_documents(docs)
    answer = chat(vector_store)

    st.text(answer)


if __name__ == "__main__":
    main()
