# PKGS
# imprting the required function
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from functions import plot_word, most_mentioned_people, text_analyzer, entity_analyzer, text_fit, sumy_summarizer,insert_data
# DATA BASE PKGS
import plotly.express as px
import mysql.connector
# DATA ANALYSIS PKGS
import pandas as pd
import numpy as np
import seaborn as sns
# VISUALIZATION PKGS
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# STREAMLIT PKGS
import streamlit as st
# NLP PKGS
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
# ml pkgs
# PKGS
c = CountVectorizer(stop_words='english')

# getting the data fom the datbases
# creating the database connection
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="alumnaes"
# )
# # quering the database
alumnaes_data=pd.read_csv('modified.csv')
# alumnaes_data = pd.read_sql("""
#             SELECT *
#             FROM alumnaes
#             """, con=mydb)
# # systems admins data
# admins_data = pd.read_sql("""
#             SELECT *
#             FROM admins
#             """, con=mydb)


def main():
    # heqader section of the system
    st.title("AMLUMNAES OPINIOIN MINING SYSTEM NLP")
    st.image('images/nlp.png')
    menu = ['alumnae review','Admin Pannel', 'home'] #
    choice = st.sidebar.selectbox('menu', menu)
    # admin login panel
    if choice == 'Admin Pannel':

        st.header('summary data (first 10 raws)')
        st.write(alumnaes_data.head())
        st.header('summary statistics')
        st.write(alumnaes_data.describe())
        st.header('summary data (last 10 raws)')
        st.write(alumnaes_data.tail())
        st.header('summary data (information size)')
        col1 = alumnaes_data.shape[1]
        col0 = alumnaes_data.shape[0]
        st.success('The data has {} Columns with {} Records'.format(col1, col0))
        menu = ['Sentiment Analysis']
        choice = st.sidebar.selectbox('Analysis Section', menu)
        #  working with admin choices
        if choice == 'Sentiment Analysis':
            st.header('select course you would like to view analysis')
            course = alumnaes_data['course'].unique().tolist()
            year = sorted(alumnaes_data['grad_year'].unique())
            col1, col2 = st.columns(2)
            col1.header("Course Selection")
            col2.header("Year Selection")
            selected_courses = col1.multiselect(
                'which courses do you want to analyse', course)
            selected_year = col2.multiselect(
                'which year do you want to analyse', year)
            selected_data = alumnaes_data[alumnaes_data['course'].isin(
                selected_courses)]
            data = selected_data[selected_data['grad_year'].isin(
                selected_year)]
            if st.checkbox('use all data'):
                data = alumnaes_data
            st.success('you have selected  {} courses for the with a total records of {}'.format(
                len(selected_courses), len(selected_data)))
            # Tokenization
            st.subheader('REVIEWS SECTION')
            if st.checkbox("Show Tokens and Lemma"):
                st.subheader("Tokenizing selected data Your Text ...... ")
                nlp_result = text_analyzer(str(data.review))
                st.json(nlp_result)
            # end of tokenixa==zation section

                # Entity Extraction
            if st.checkbox("Show Named Entities"):
                st.subheader("showing named entities from the document.....")
                entity_result = entity_analyzer(str(data.review))
                st.json(entity_result)
                # end of named entity extraction section

                # Entity Extraction
            if st.checkbox("Show Most Mentioned"):
                st.subheader("showing most mentioned words.....")
                mostment = most_mentioned_people(data)
                st.write(mostment)
                st.subheader("showing most entioned words....")

                # plotting most mentioned words
                fig = px.bar(mostment, y="count", x="text")
                st.plotly_chart(fig)
                # plotting the data world Cloud
                st.write(data.import_units.head())
                data=pd.DataFrame(data)
                st.subheader('review visualization')
                plot_word(data['review'])
                st.subheader('unit relevant visualization')
                plot_word(data['import_units'])
                # end of named entity extraction section

                # Sentiment Analysis
            if st.checkbox("Show Sentiment Analysis"):
                st.subheader("Analyse Your Text")
                blob = TextBlob(str(data.review))
                result_sentiment = blob.sentiment
                st.success(result_sentiment)
                st.write(text_fit(data, c, LogisticRegression(), 1))
        # SENTIMENT ANALYSIS SECTION END

                # Summarization
            if st.checkbox("Summarization Section"):
                st.subheader("Summarizing  Your document")
                summary_options = st.selectbox(
                    "Choose Summarizer", ['sumy', 'gensim'])
                if st.button("Summarize"):
                    if summary_options == 'sumy':
                        st.text("Using Sumy Summarizer ..")
                        summary_result = sumy_summarizer(str(data.review))
                        st.text("Using Gensim Summarizer ..")
                        summary_result = summarize(str(data.review))
                    else:
                        st.warning("Using Default Summarizer")
                        st.text("Using Gensim Summarizer ..")
                        summary_result = summarize(str(data.review))

                    st.success(summary_result)
            # summarization END

            else:
                st.warning('KINDLY MAKE SURE TO SELECT ATLEAST ONE COURSE')
            # st.write(data)

    # ALUMNAES/DTUDENTS
    #  home section
    if choice == 'home':
        st.subheader('HOME')
    # form feed section
    if choice == 'alumnae review':
        st.subheader('WELCOME TO  NLP AMLUMNAES OPINIOIN MINING SYSTEM')
        # ?arcademic form handling sector
        with st.form(key='form1'):
            st.header('Arcademic Section')
            course = st.text_input(label='Enter the course Undertaken')
            grad_year = st.number_input(
                min_value=2014, max_value=2020, step=1, label='what year did you graduated??')
            review =st.text_area('Review the course')
            rating = st.number_input(
                min_value=0, max_value=5, step=1, label='rate course undertaken')
            if rating>2:
                polarity =1
            else:
                polarity = 0
        # work details Section
            st.header('Job/ Work  Section')
            current_position = st.text_input(
                label='whats your current position of Job?')
            
            important_units = st.text_input(
                    label='what units do you find in handy with your current position as a ?')
      
            experienced_years = st.number_input(
                    min_value=0, max_value=5, step=1, label='how many years have you in the field as a so far')
            val = (course,current_position, review, rating, grad_year,important_units,experienced_years, polarity)
            submit = st.form_submit_button(label='submit arcademic section')       

            if submit:
                insert_data(val)
                st.write('YOPU HAVE SUBMITTEDE THE FOLLOWING DATA\n',val)
                st.success('thank you very much for your valuable reviews')
                st.balloons()

            # commmmit the data into the database


st.sidebar.subheader("About App")
st.sidebar.text("NLP ALUMNAES  OPINION MINING SYSTEM")
st.sidebar.info("Deliverying Analytical products in a more consumable manner")


st.sidebar.subheader("By")
st.sidebar.text("Kiplangat David Kipngeno)")
st.sidebar.text("Junior Machine Learning Engeneer")


if __name__ == '__main__':
    main()
