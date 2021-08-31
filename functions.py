# requireres pkgs
from gensim.summarization.summarizer import summarize
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.summarization import keywords
from nltk.tokenize import sent_tokenize
from sklearn.dummy import DummyClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
import nltk
import numpy as np
import seaborn as sns
import pandas as pd
import streamlit as st
import mysql.connector
import plotly.express as px
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
# database pkgs
# streamlit pkgs
# import spacy
# PKGS
st.set_option('deprecation.showPyplotGlobalUse', False)
# machine learning pckages for POLARITY ANALYSIS
# PKGS
c = CountVectorizer(stop_words='english')
# summay text summarization pkgs
# Sumy Summary Pkg

# text analyzer


@st.cache
def text_analyzer(my_text):
    nlp = en_core_web_sm.load()
    docx = nlp(my_text)
    # tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(
        token.text, token.lemma_))for token in docx]
    return allData
# Function for Sumy Summarization

# text summarizer


@st.cache
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

# entity analyzer


@st.cache
def entity_analyzer(my_text):
    nlp = en_core_web_sm.load()
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_)for entity in docx.ents]
    allData = ['"Token":{},\n"Entities":{}'.format(tokens, entities)]
    return allData

# getting most common mentioned people/lecturers


@st.cache
def most_mentioned_people(data):
    tokens = nlp(''.join(str(data.review.tolist())))
    person_list = []

    for ent in tokens.ents:
        if ent.label_ == 'PERSON':
            person_list.append(ent.text)

    person_counts = Counter(person_list).most_common(20)
    df_person = pd.DataFrame(person_counts, columns=['text', 'count'])
    return df_person
# data visualizations functions


@st.cache
def plot_worldcloud(data):
    comment_words = ''
    stopwords = set(STOPWORDS)
    # iterate through the csv file
    for val in data.review:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)
    # plot the WordCloud image
    plt.figure(figsize=(4, 4), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot()

# data cleaning funtions
# 434 review rows to clean


@st.cache
def clean_text(words):
    corpus = []
    for i in range(1, len(words)):
        # column : "Review", row ith
        review = re.sub('[^a-zA-Z]', ' ', data['review'][i])
        # convert the reviews to lower cases
        review = review.lower()
        # apply split(default delimiter is " ")
        review = review.split()
        # creating a poster object totake main sterm of each word
        # instatiate the porterstemmer
        ps = PorterStemmer()
        # stermming each woird by loopping through each pof the strings
        review = [ps.stem(word) for word in review if not word in set(
            stopwords.words('english'))]
        # lets rejoiun back the string array elemnts to create inmto a string

        review = ' '.join(review)
        # append each  strting now to the corpus array of clean text
        corpus.append(review)
    return corpus
# MACHINE LEARNING FUNCTION FOR TATEXT ANALYSIS
# create a function


@st.cache
def text_fit(data, nlp_model, ml_model, coef_show=1):
    X = data['review']
    y = data['polarity']
    X_c = nlp_model.fit_transform(X)
    st.write('features: {}'.format(X_c.shape[1]))
    X_train, X_test, y_train, y_test = train_test_split(X_c, y)
    st.write(' train records: {}'.format(X_train.shape[0]))
    st.write(' test records: {}'.format(X_test.shape[0]))
    ml = ml_model.fit(X_train, y_train)
    acc = ml.score(X_test, y_test)
    st.write('Model Accuracy: {}'.format(acc))

    if coef_show == 1:
        w = nlp_model.get_feature_names()
        coef = ml.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word': w, 'Coefficient': coef})
        coeff_df = coeff_df.sort_values(
            ['Coefficient', 'Word'], ascending=[0, 1])
        st.write('\n')
        st.write('-Top 20 positive-')
        st.write(coeff_df.head(20))
        fig = px.bar(coeff_df.head(20), y="Coefficient", x="Word")
        st.plotly_chart(fig)
        st.write('\n')
        st.write('-Top 20 negative-')
        st.write(coeff_df.tail(20))
        fig = px.bar(coeff_df.tail(20), y="Coefficient", x="Word")
        st.plotly_chart(fig)


# database functons
# wodcloud ploting function rectified from previous update
@st.cache
def plot_word(data):
    comment_words = ''
    stopwords = set(STOPWORDS)

    # iterate through the csv file
    for val in data:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot()

# plot wordcloud end


@st.cache
def insert_data(val):
    # creating a connection to db
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="alumnaes"
    )
    mycursor = mydb.cursor()
    sql = "INSERT INTO alumnaes (course,current_possion,review,rating,grad_year,import_units,experienced_years,polarity) \
                VALUES (%s, %s,%s,%s,%s,%s,%s,%s)"
    mycursor.execute(sql, val)
    mydb.commit()
