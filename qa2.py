import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from docx import Document
from typing import List, Tuple, Dict
import chromadb

client = chromadb.PersistentClient(path="/db")

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model='gpt-3.5-turbo-16k',
        temperature=0.5,
        max_tokens=12000,
       # messages=[{"role": "system", "content": "You are a marketer and sales analyst"}]
    # Add any additional model parameters if needed
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def get_relevant_documents_and_metadata(query: str, vectorstore) -> Tuple[List[str], List[Dict]]:
    """Get relevant documents and their metadata from Chroma db using query."""
    results = vectorstore.query(query_texts=query, include=["metadatas", "documents"])
    
    # Extract documents and metadatas from results
    documents = [doc['document'] for doc in results['documents']]
    metadatas = [meta['metadata'] for meta in results['metadatas']]
    
    return documents, metadatas

def handle_userinput(user_question):
    # Get relevant documents and their metadata
    documents, metadatas = get_relevant_documents_and_metadata(user_question, st.session_state.vectorstore)
    
    # Here, you can format the documents and metadatas as needed and send them to the chat as a system message
    system_message = format_documents_and_metadata(documents, metadatas)
    st.write(system_message)
    
    # Continue with your existing chatbot logic
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Чат с информацией из всех файлов сразу",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Задайти вопрос ИИ по всем документам сразу что бы получить ответ по тематике из всех файлов:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
