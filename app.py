from data import load_docs, index_documents, chat

def main():
    import streamlit as st

    st.title("Modal Chatbot")

    if st.button('Reindex Code'):
        st.text("Reindexing")
        st.session_state.vector_store = index_documents(True)
        st.rerun()
        
    if "vector_store" not in st.session_state:
        st.text("Loading Chatbot...")
        st.session_state.vector_store = index_documents(False)
        st.rerun()
    else:
        vector_store = st.session_state.vector_store

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask the Modal Chatbot!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                result = chat(vector_store, prompt, st.session_state.chat_history)
                answer = result["answer"]
                st.markdown(answer) 
                st.session_state.chat_history.append((prompt, answer))
            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
