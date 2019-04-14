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
from sklearn.model_selection import train_test_split
import numpy
import sys

numpy.set_printoptions(threshold=sys.maxsize)

seed = 7
numpy.random.seed(seed)


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
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
 
# load reviews
in_filename = "review.dat"
doc = load_doc(in_filename)

# prepare the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create sequences
#encoded = tokenizer.texts_to_sequences([doc])[0]
sequences = list()

for line in doc.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(2, len(encoded)):
        print(line)
        print(encoded[:])
        sequence = encoded[i-2:i+1]
        print(sequence)
        sequences.append(sequence)
    for i in range(3, len(encoded)):
        sequence = encoded[i-3:i+1]
        sequences.append(sequence)
    if len(encoded) > 4:
        for i in range(4, len(encoded)):
            sequence = encoded[i-4:i+1]
            sequences.append(sequence)
    if len(encoded) > 5:
        for i in range(5, len(encoded)):
            sequence = encoded[i-5:i+1]
            sequences.append(sequence)

#sys.exit()

print('Total Sequences: %d' % len(sequences))

print(sequences[1])
print(sequences[2])
print(sequences[3])
print(sequences[4])
print(sequences[5])
print(sequences[6])
print(sequences[7])
print(sequences[8])
print(sequences[9])
print(sequences[10])
print(sequences[11])
print(sequences[12])
print(sequences[13])
print(sequences[14])
print(sequences[15])
print(sequences[16])

#sys.exit()


max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

# assign input/output elements for sequences
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]

print(X[1])
print(y[1])

y = to_categorical(y, num_classes=vocab_size)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#print(X_train)
#print(y_train)


# create model with LSTM layer with 128 memory units
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length-1))
model.add(LSTM(256))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=2)
model.fit(X, y, validation_split=0.2, batch_size=64, epochs=20, verbose=2)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#cvscores.append(scores[1] * 100)
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

model.save('lstm.h5')


