import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from docx import Document

load_dotenv()

def read_file(file):
    if file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(file)
        return df.to_string(index=False)
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif file.type == "application/pdf":
        text = ""
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
        return text
    elif file.type == "text/plain":
        return file.getvalue().decode()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    else:
        return None

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

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Загрузите несколько документов разных форматов и нажмите 'Process'", accept_multiple_files=True, type=["pdf", "csv", "txt", "xlsx"])
        if st.button("Process"):
            with st.spinner("Processing"):
                texts = []
                for file in uploaded_files:
                    raw_text = read_file(file)
                    if raw_text:  # проверяем, что текст был успешно извлечен
                        texts.append(raw_text)
            
                # объединяем все тексты
                combined_text = "\n".join(texts)

                # get the text chunks
                text_chunks = get_text_chunks(combined_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
