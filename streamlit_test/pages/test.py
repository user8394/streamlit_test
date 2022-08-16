import streamlit as st
from transformers import pipeline

load_pipeline = st.cache(pipeline)



st.title('Test app page 2')


st.write('Loading model...')

#model = load_pipeline('zero-shot-classification')

st.write('Done ✅')
