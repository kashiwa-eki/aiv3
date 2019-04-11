from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import random

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

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

#with open(in_filename, "r") as f:
#    doc = f.read().split('\n')


# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create line-based sequences
sequences = list()
for line in doc.split('\n'):
	encoded = tokenizer.texts_to_sequences([line])[0]
	sequence = encoded[:len(encoded)-1]
	sequences.append(sequence)
	sequence = encoded[:]
	sequences.append(sequence)
#	for i in range(1, len(encoded)):
#		sequence = encoded[:i+1]
#		sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
#out_filename = 'review_sequences.txt'
#save_doc(sequences, out_filename)

#print(sequences)

# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

print(sequences)

# save sequences to file
#out_filename = 'review_sequences.txt'
#save_doc(sequences, out_filename)

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(X, y, epochs=10, verbose=2)
model.save('lstm.h5')

#evaluate model
print(generate_seq(model, tokenizer, max_length-1, 'good sounding strings and they', 2))
print(generate_seq(model, tokenizer, max_length-1, 'unwound strings were a tiny bit', 2))

