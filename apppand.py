import streamlit as st
import pandas as pd
import os
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

# read excel files from directory
#excel_dir = "/home/myhome202323/ask-multiple-pdfs/data/"
#excel_files = [os.path.join(excel_dir, f) for f in os.listdir(excel_dir) if f.endswith(".xlsx")]

# create DataFrame from excel files
#df = pd.concat([pd.read_excel(f) for f in excel_files])
file_path = '/home/myhome202323/ask-multiple-pdfs/data/krossovki.xlsx'
df = pd.read_excel(file_path)
#df = from_excel(file_path)
# get the list of column names from the DataFrame
#column_names = df.columns.tolist()

# create a selectbox for the user to choose a column
#selected_column = st.selectbox('Select a column', column_names)

# get the list of unique values from the selected column in the DataFrame
#options = df[selected_column].unique().tolist()

# create a text input field for the user to enter their search query
#search_query = st.text_input('Search')

# filter the list of options based on the search query
#filtered_options = [option for option in options if search_query.lower() in str(option).lower()]

# create a selectbox to display the filtered options
#selected_option = st.selectbox('Select an option', filtered_options)

# create OpenAI LLM instance
#llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), temperature=0)
# create PandasAI instance
#pandas_ai = PandasAI(llm=llm)
#pandas_ai = PandasAI(llm=llm, conversational=True, save_charts=True, enable_cache=False)
pandas_ai = PandasAI(llm, verbose=True, conversational=True, enable_cache=False)

from io import BytesIO
from PIL import Image

def handle_userinput(user_question):
    # use PandasAI to process user question
    response = pandas_ai.run(df, prompt=user_question)

    # check if the response is a DataFrame
    if isinstance(response, pd.DataFrame):
        # display the DataFrame using st.dataframe
        st.dataframe(response)
    elif isinstance(response, pd.Series):
        # display the Series using st.line_chart
        st.line_chart(response)
    elif isinstance(response, plt.Figure):
        # display the Matplotlib figure using st.pyplot
        st.pyplot(response)
    elif isinstance(response, str):
        # display the string as text using st.write
        st.write(response)
    elif isinstance(response, dict):
        # display the dictionary as a table using st.table
        st.table(response)
    elif isinstance(response, list):
        # display the list as text using st.write
        st.write(response)
    elif isinstance(response, bytes):
        # create a BytesIO object from the byte string
        byte_stream = BytesIO(response)
        # open the image and display it using st.image
        img = Image.open(byte_stream)
        st.image(img)
    else:
        # if the response type is not recognized, display a message
        st.write("Неизвестный тип данных.")

def main():
    st.set_page_config(page_title="Анализ продаж на Wildberries:", page_icon=":books:")
    st.header("Проанализируй конкурентов и продажи на Wildberries:books:")
    user_question = st.text_input("Задай вопрос на тему товара:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
