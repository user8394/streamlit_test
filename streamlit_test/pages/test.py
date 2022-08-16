import streamlit as st
from transformers import pipeline

load_pipeline = st.cache(pipeline)



st.title('Test app page 2')


st.write('Loading model...')

zero_shot_classifier = load_pipeline('zero-shot-classification')


class ZeroShotClassification:

	def __init__(self, classifier=None):
		self.classifier = classifier or pipeline('zero-shot-classification')


	def __call__(self, sentence):
		return self.predict(sentence)


	def get_label_score(self, sentence, label, labels):

		result = self.classifier(sentence, labels)
		label_index = result['labels'].index(label)
		label_score = result['scores'][label_index]

		return label_score


	def predict(self, sentence):

		# negativity_score = self.get_label_score(sentence, 'negative', [
		# 	'negative',
		# 	'positive',
		# 	'neutral',
		# ])


		# if negativity_score < 0.7:
		# 	return (False, None)


		return self.classifier(sentence, [
			'suicide',
			'sexual abuse',
			'physical abuse',
			'self harm',
      'violence',
      'feeling unsafe',
			'other'
		])
  
 

st.write('Done ✅')

cls = ZeroShotClassifier(zero_shot_classifier)

st.write(cls('I want to hurt myself'))
