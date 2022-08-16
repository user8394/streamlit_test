import streamlit as st
from transformers import pipeline
import plotly.figure_factory as ff
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
	zero_shot_classifier = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-1')

st.success('Done ✅')


class ZeroShotClassification:

	def __init__(self, classifier=None):
		self.classifier = classifier or pipeline('zero-shot-classification')


	def __call__(self, *args):
		return self.predict(*args)


	def get_label_score(self, sentence, label, labels):

		result = self.classifier(sentence, labels)
		label_index = result['labels'].index(label)
		label_score = result['scores'][label_index]

		return label_score


	def predict(self, sentence, labels):

		# negativity_score = self.get_label_score(sentence, 'negative', [
		# 	'negative',
		# 	'positive',
		# 	'neutral',
		# ])


		# if negativity_score < 0.7:
		# 	return (False, None)


		return self.classifier(sentence, labels)
  
 
cls =  ZeroShotClassification(zero_shot_classifier)


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
	],
	[
		'depression',
		'violence',
		'physical abuse',
		'sexual abuse',
		'feeling unsafe',
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

with st.expander('View results...'):
	
	st.write('#### 1. Sentiment Analysis')
	
	positive = int(sentiment['scores'][sentiment['labels'].index('positive')] * 100)
	negative = int(sentiment['scores'][sentiment['labels'].index('negative')] * 100)
	neutral = int(sentiment['scores'][sentiment['labels'].index('neutral')] * 100)

	col1, col2, col3 = st.columns(3)
	col1.metric("Positivity", str(positive) + '%')
	col2.metric("Negative", str(negative) + '%')
	col3.metric("Neutral", str(neutral) + '%')


	st.write('#### 2. Zero Shot Classification')

	hist = []

	for label in classification_labels:
		score = classification['scores'][classification['labels'].index(label)]
		hist.append(score)

	st.bar_chart({'data': hist})
	
	st.write('Raw output:')
	st.write(classification)
