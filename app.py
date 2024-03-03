from data import load_docs, index_documents, chat

def main():
    import streamlit as st

    st.title("Modal Chatbot")

    if st.button('Reindex Code'):
        print("CLICKED")
        
    vector_store = index_documents(false)
    answer = chat(vector_store)

    st.text(answer)


if __name__ == "__main__":
    main()
