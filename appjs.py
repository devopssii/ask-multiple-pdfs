import streamlit as st
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import openai
import os
from dotenv import load_dotenv

load_dotenv()

def get_excel_text(excel_docs):
    text = ""
    for excel in excel_docs:
        df = pd.read_excel(excel)
        text += df.to_string(index=False)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model='gpt-3.5-turbo-16k',
        temperature=0.5,
        max_tokens=12000,
       # messages=[{"role": "system", "content": "You are a marketer and sales analyst >
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

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# read excel files from directory
excel_dir = "/home/myhome202323/ask-multiple-pdfs/data/"
excel_files = [os.path.join(excel_dir, f) for f in os.listdir(excel_dir) if f.endswith(".xlsx")]

# get excel text
raw_text = get_excel_text(excel_files)

# get the text chunks
text_chunks = get_text_chunks(raw_text)

# create vector store
vectorstore = get_vectorstore(text_chunks)

# create conversation chain
conversation_chain = get_conversation_chain(vectorstore)

def main():
    st.set_page_config(page_title="Задай вопрос на тему продаж на Wildberries", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = conversation_chain
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Задай вопрос на тему продаж на Wildberries он найдет информацию о любом товаре и расскажет :books:")
    user_question = st.text_input("Напиши 5 моделей кроссовок с продажами от 1000:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
