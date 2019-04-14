from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
import random

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text
 
# source text
in_filename = "review.dat"
doc = load_doc(in_filename)

# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])


# load model
model = load_model('lstm.h5')

#evaluate model
otext = "however the manual shows exact settings for computer use"
seed_text = "however the manual shows exact settings for"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "the bass is tight and articulate which is what you want"
seed_text = "the bass is tight and articulate which is what"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "not big enough for a strap though with everything else in there"
seed_text = "not big enough for a strap though with everything else"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "ive been using daddario xl nickel wounds on my pbass for a long time"
seed_text = "ive been using daddario xl nickel wounds on my pbass for a"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "it also seem tightly put together at the ends"
seed_text = "it also seem tightly put together at"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "ive purchased several switchcraft jacks and they just work"
seed_text = "ive purchased several switchcraft jacks and they"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "however i could have got this product for a lot cheaper"
seed_text = "however i could have got this product for a"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "no buzzing on the strings when i put it on"
seed_text = "no buzzing on the strings when i put"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "these are the best ones ive tried so far"
seed_text = "these are the best ones ive tried"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')

otext = "the sound is very big full and nice"
seed_text = "the sound is very big full"
print("original: " + otext + '\n')
print("predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
print('\n')



