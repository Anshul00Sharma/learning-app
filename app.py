## RAG Q&A Conversation With PDF Including Chat History
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from streamlit_chat import message

import streamlit as st


# Initialize session state
if 'screen' not in st.session_state:
    st.session_state.screen = "upload"
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
HF_TOKEN = st.sidebar.text_input("Enter your HF_TOKEN:", type="password")

# Function to render the upload screen
def upload_screen():
    st.title("Upload PDFs and Set API Keys")
    # Input API keys
    st.sidebar.header("API Keys")
    
    
    # Upload PDFs
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    # Process PDFs and create vector store
    if uploaded_files and api_key and HF_TOKEN:
    
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        

        ## chat interface

        session_id=st.text_input("Session ID",value="default_session")
        ## statefully manage chat history

        if 'store' not in st.session_state:
            st.session_state.store={}

        # uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
        ## Process uploaded  PDF's
        if uploaded_files:
            documents=[]
            for uploaded_file in uploaded_files:
                temppdf=f"./temp.pdf"
                with open(temppdf,"wb") as file:
                    file.write(uploaded_file.getvalue())
                    file_name=uploaded_file.name

                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                documents.extend(docs)

        # Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            # vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=f"vectorstore/{session_id}")
               
        if st.button("Chat with PDF"):
            st.session_state.screen = "chat"
            st.rerun()
    else:
        st.warning("Please upload PDFs and enter API keys.")

# Function to render the chat screen
def chat_screen():
    st.title("Chat with PDFs")
    session_id=st.text_input("Session ID",value="default_session")
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        llm = ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It" ) 
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question

        # Answer question
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if prompt := st.chat_input():
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {"session_id":session_id}
                    },  # constructs a key "abc123" in `store`.
                )
            msg = response['answer']
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
    

# Render the appropriate screen based on session state
if st.session_state.screen == "upload":
    upload_screen()
elif st.session_state.screen == "chat":
    chat_screen()








