import pickle
import nltk
import streamlit as st
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))


vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
model_path = os.path.join(script_dir, 'model.pkl')

tfidf = pickle.load(open(vectorizer_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    transformed_sms = tfidf.transform([transform_text(input_sms)])
    prediction = model.predict(transformed_sms)[0]

    if prediction == 1:
        st.error('Spam')
    else:
        st.success('Not Spam')

print(hasattr(tfidf, 'idf_'))