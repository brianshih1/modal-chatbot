

## Building a Modal Chatbot with Modal

The goal of this project is to build a chatbot for Modal using Modal. Here is a quick demo:


https://github.com/brianshih1/modal-chatbot/assets/47339399/1042b5fb-8f7a-4b6b-b04f-182974621f09


### Data Ingestion

The data is collected from these two Modal github repos: [modal-client](https://github.com/modal-labs/modal-client) and [modal-examples](https://github.com/modal-labs/modal-examples). These are also the two repos that Modal's official docs are based off of. To generate documents from the `github repo`, I used `langchain`'s `GitLoader` and `PythonCodeTextSplitter` to turn a git repo into chunks of text.  

### Create and store Embeddings

I used the `HuggingFace` sentence-transformers for this project to generate embeddings from the chunks of text. To avoid having to regenerate the embeddings every time the application is run, I use a `Modal Volume` to store the embeddings.

### Construct Prompt

Next, we construct the prompt by first querying the Vector Store (I use `FAISS` for thie project) for relevant documents then bundling them with the user's question. We also send chat history as part of the prompt. Finally, we send the prompt to `OpenAI's GPT` to get a response. These steps are abstracted with `langchain`'s `ConversationalRetrievalChain`. 

### Web App

The web app is built with streamlit and deployed with Modal.
