import streamlit as st
import pandas as pd
import os
import pandas as pd
import pandasai as pdai
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.llm.starcoder import Starcoder
from pandasai.llm.open_assistant import OpenAssistant
from pandasai.llm.google_palm import GooglePalm

# read excel files from directory
excel_dir = "data/"
excel_files = [os.path.join(excel_dir, f) for f in os.listdir(excel_dir) if f.endswith(".xlsx")]

# create DataFrame from excel files
df = pd.concat([pd.read_excel(f) for f in excel_files])

# create OpenAI LLM instance
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# create PandasAI instance
pandas_ai = PandasAI(llm=llm)

def handle_userinput(user_question):
    # use PandasAI to process user question
    response = pandas_ai.ask(df, prompt=user_question)

    # display response
    st.write(response)

def main():
    st.set_page_config(page_title="Chat with multiple Excel files", page_icon=":books:")
    st.header("Chat with multiple Excel files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
