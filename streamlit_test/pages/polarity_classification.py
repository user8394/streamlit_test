import streamlit as st
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import os


st.title('Polarity Classification')
st.write('''
This model uses spaCy to predict the polarity (degree of positive or negative emotion) within a piece of text. This can be used to flag potentially extreem of concerning content 
within user responses. The advantage of this model over NLI is simplicity and inference speed.
''')

st.spinner('Downloading corpus...')
os.system('python -m spacy download en_core_web_sm')
st.success('Done ✅')
st.spinner('Loading model...')
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')
st.success('Done ✅')


def polarity(text):
  return nlp(text)._.polarity

input = st.textarea('Input sentence')

with st.spinner('Analysing...'):
  score = polarity(input)

st.success('Analysing complete ✅')
st.write('#### Result')
st.metric("Polarity", str(score) + '%')

