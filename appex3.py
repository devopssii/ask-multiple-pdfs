import streamlit as st
import openai
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from transformers import pipeline
import requests
import json
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

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
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Excel files",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    class GPT35Turbo:
        def __init__(self):
            self.model = "gpt-3.5-turbo"
            self.api_key = os.getenv("OPENAI_API_KEY")
            openai.api_key = self.api_key

        def generate(self, prompt):
            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                max_tokens=512,
                temperature=0.5
            )
            return response.choices[0].text.strip()

    def get_conversation_chain(vectorstore):
        llm = GPT35Turbo()

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
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

    st.header("Chat with multiple Excel files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        excel_docs = st.file_uploader(
            "Upload your Excel files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get excel text
                raw_text = get_excel_text(excel_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()
