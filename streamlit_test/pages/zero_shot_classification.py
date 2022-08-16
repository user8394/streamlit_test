import streamlit as st
from transformers import pipeline
import numpy as np


#col1, col2 = st.columns(2)

st.title('Zero Shot Classification Model 🔮')

st.write('''
This model uses a pretrained NLI pipeline to classify a sentence amoung a set of given labels. This can be directly applied to the problem of flagging safe-guarding issues 
by providing a number of short safe guarding labels such as "physical abuse" or "self harm." For each label, the model assigns a score between 0 and 1 representing how 
confident it is that a given sentence falls into that category. 
''')

with st.expander('More information...'):
	
	st.write('''
	### Advantages

	The `zero-shot-classification` pipeline leverages a LLM giving it a good understanding of contextual cues and implicit meanings of sentences. Labels can be added 
	or removed with any additional training, meaning the system can be maintained easily. 

	### Disadvantages

	The model is rather large at ~999MB so can't be easily run on limited hardware. However this shouldn't be a problem when deploying to cloud providers like GCP.


	### 💻 Usage

	To test out the model, simply enter a sentence and press "classify." The model whill produce a distribution over the predicted label. You can also edit the 
	specific labels. Have fun 🤗!
	''')

with st.spinner('🧠 Loading NLI model (`valhalla/distilbart-mnli-12-1`)...'):
	try:
		del cls
	except:
		pass
	
	cls = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-1')

st.success('Done ✅')
  

classification_labels = st.multiselect(
     'Chosen classification labels...',
	[
		'self harm',
		'depression',
		'suicide',
		'physical abuse',
		'sexual abuse',
		'sexual harrassment',
		'feeling unsafe',
		'violence',
		'other',
		'bullying',
		'addiction',
		'drug abuse',
	],
	[
		'self harm',
		'physical abuse',
		'sexual abuse',
		'feeling unsafe',
		'addiction',
		'depression',
		'other',
	]
)

sentence = st.text_area('Input text...', '''
I feel unhappy :(
     ''')

with st.spinner('Analysing...'):

	sentiment = cls(sentence, ['positive', 'negative', 'neutral'])
	classification = cls(sentence, classification_labels)

st.success('Analysis complete ✅')

with st.expander('View results summary...'):
	
	st.write('#### ❤ Sentiment Analysis')
	
	positive = int(sentiment['scores'][sentiment['labels'].index('positive')] * 100)
	negative = int(sentiment['scores'][sentiment['labels'].index('negative')] * 100)
	neutral = int(sentiment['scores'][sentiment['labels'].index('neutral')] * 100)

	col1, col2, col3 = st.columns(3)
	col1.metric("Positivity", str(positive) + '%')
	col2.metric("Negativity", str(negative) + '%')
	col3.metric("Neutrality", str(neutral) + '%')
	
	
	st.write('---')

	st.write('#### ⚖ Zero Shot Classification')

	hist = []

	for label in classification_labels:
		score = classification['scores'][classification['labels'].index(label)]
		hist.append(score)

	st.bar_chart({'data': hist})
	
	st.write(f'Most likely: {classification_labels[np.argmax(hist)]}')
	

with st.expander('View raw output...'):
		
	st.write('sentiment_analysis: ')
	st.write(sentiment)
	st.write('zero_shot_classification: ')
	st.write(classification)
