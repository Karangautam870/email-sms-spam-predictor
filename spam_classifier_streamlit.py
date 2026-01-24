import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


current_dir = os.path.dirname(os.path.abspath(__file__))


tfidf_path = os.path.join(current_dir, 'vectorizer.pkl')
model_path = os.path.join(current_dir, 'model.pkl')

try:
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Model files not found. Please ensure vectorizer.pkl and model.pkl are in the same directory as this script.")
    st.stop()

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
    ps = PorterStemmer()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y) 

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if not input_sms.strip():
    st.warning('Please enter a message to classify.')
else:
    if st.button('Predict'):
        transformed_sms = tfidf.transform([transform_text(input_sms)])

        prediction = model.predict(transformed_sms)[0]
        if prediction == 1:
            st.error('Spam')
        else:
            st.success('Not Spam')

