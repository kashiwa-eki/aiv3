#  AI Postmodule Assignment - Predict last two words of review
#  MSBA Spring 2019
#  April 29, 2019
#  Study Group 17
#  - Madi Benjamin
#  - Jason Inzer
#  - Shivika Mehta
#  - Ann Michael
#
#  Code adapted from multiple LSTM examples found online
#  Special thanks to: https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/


from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import random
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

seed = 7
numpy.random.seed(seed)


# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words (2 for our example)
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
 
# source data
in_filename = "review.dat"
doc = load_doc(in_filename)

# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# convert text to integers for sequences
encoded = tokenizer.texts_to_sequences([doc])[0]
sequences = list()
# generate sequence 2 words in -> 1 word out
for i in range(2, len(encoded)):
    sequence = encoded[i-2:i+1]
    sequences.append(sequence)
# generate sequence 3 words in -> 1 word out
for i in range(3, len(encoded)):
    sequence = encoded[i-3:i+1]
    sequences.append(sequence)
# generate sequence 4 words in -> 1 word out
for i in range(4, len(encoded)):
    sequence = encoded[i-4:i+1]
    sequences.append(sequence)
# generate sequence 5 words in -> 1 word out
for i in range(5, len(encoded)):
    sequence = encoded[i-5:i+1]
    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))

# take care of variable length sequences by adding padding
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

# assign sequences to input/output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# create model with single hidden LSTM layer with 128 memory units
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length-1))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model and evaluate using accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model using 80% of data for training and 20% for testing. 100 epochs and batch size of 64.
model.fit(X, y, validation_split=0.2, batch_size=64, epochs=100, verbose=2)

# save model
model.save('lstm.h5')


