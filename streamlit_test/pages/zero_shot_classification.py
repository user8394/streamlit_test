import streamlit as st
from transformers import pipeline
import plotly.figure_factory as ff
import numpy as np



st.title('Zero Shot Classification Model 🔮')

st.write('''

## About

This model uses a pretrained NLI pipeline to classify a sentence amoung a set of given labels. This can be directly applied to the problem of flagging safe-guarding issues 
by providing a number of short safe guarding labels such as "physical abuse" or "self harm." For each label, the model assigns a score between 0 and 1 representing how 
confident it is that a given sentence falls into that category. 

## Advantages

The `zero-shot-classification` pipeline leverages a LLM giving it a good understanding of contextual cues and implicit meanings of sentences. Labels can be added 
or removed with any additional training, meaning the system can be maintained easily. 

## Disadvantages

The model is rather large at ~999MB so can't be easily run on limited hardware. However this shouldn't be a problem when deploying to cloud providers like GCP.


## Usage

To test out the model, simply enter a sentence and press "classify." The model whill produce a distribution over the predicted label. You can also edit the 
specific labels. Have fun 🤗!
''')


st.write('🧠 Loading model (valhalla/distilbart-mnli-12-1)...')


zero_shot_classifier = pipeline('zero-shot-classification', model='valhalla/distilbart-mnli-12-1')


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
  
 

st.write('Done ✅')

@st.cache
def create_model():
	return ZeroShotClassification(zero_shot_classifier)

cls = create_model()


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

st.write('[+] Performing sentiment analysis...')
	 
sentiment = cls(sentence, ['positive', 'negative', 'neutral'])

st.write('[+] Performing zero shot classification...')

classification = cls(sentence, classification_labels)
	 
st.write('[+] Done ✅')

st.write(sentiment)
st.write(classification)
