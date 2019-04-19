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
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
import numpy as np
import random
import sys

# set seed for random number generator
seed = 7
np.random.seed(seed)

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

# save doc
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
        
# return words from sequence
def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

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
count = len(open(in_filename).readlines())
print(count)
#sys.exit()
numrows = round(count * 0.8)
print(numrows)

train_doc = ''
test_doc = ''

cnt = 1
with open(in_filename) as f:
    for line in f:
        #print(line)
        if cnt > numrows:
            test_doc += line
        else:
            train_doc += line
        cnt += 1

with open('train.txt', 'w') as f:
    f.write(train_doc)
with open('test.txt', 'w') as f:
    f.write(test_doc)


# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# convert text to integers for sequences
encoded = tokenizer.texts_to_sequences([doc])[0]
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
#reverse_map = dict(zip(word_to_id.values(), word_to_id.keys()))

sequences = list()

# create sequences
#for i in range(1, len(encoded)):
#    sequence = encoded[i-1:i+1]
#    sequences.append(sequence)
for i in range(2, len(encoded)):
    sequence = encoded[i-2:i+1]
    sequences.append(sequence)
for i in range(3, len(encoded)):
    sequence = encoded[i-3:i+1]
    sequences.append(sequence)
for i in range(4, len(encoded)):
    sequence = encoded[i-4:i+1]
    sequences.append(sequence)
for i in range(5, len(encoded)):
    sequence = encoded[i-5:i+1]
    sequences.append(sequence)


print('Total Sequences: %d' % len(sequences))
sys.exit()

# take care of variable length sequences by adding padding
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

# assign input/output elements for model
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]

print(len(sequences))
print(len(X))
print(len(y))
numrows = round(len(X) * 0.8)
print(numrows)
X_train = X[:numrows]
y_train = y[:numrows]
X_test = X[numrows:]
y_test = y[numrows:]

print(X_train[numrows-1])
print(y_train[numrows-1])
print(sequences[numrows-1])
print(X_test[0])
print(y_test[0])
print(sequences[numrows])
#print(list(map(sequence_to_text, X_train[numrows-1])))
#print(list(map(sequence_to_text, X_test[0])))
print(" ".join([reverse_word_map[x] for x in X_test[:10]]))

sys.exit()

# split data into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
y_train = to_categorical(y_train, num_classes=vocab_size)
y_test4cm = y_test
y_test = to_categorical(y_test, num_classes=vocab_size)

# create model with single hidden LSTM layer with 128 memory units
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length-1))
#model.add(LSTM(128, return_sequences=True))
model.add(LSTM(512))
model.add(Dropout(0.2))
#model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model and evaluate using accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model using training data. 100 epochs and batch size of 64.
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, verbose=2)

# save model
model.save('lstm.h5')

# evaluate model
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print('Test accuracy: ' + str(score[1]))

# predict the test set results
y_pred = model.predict_classes(X_test, batch_size=32, verbose=1)
print(y_pred)
print(y_test4cm)

# create a confusion matrix
cm = confusion_matrix(y_test4cm, y_pred)
print('Confusion Matrix')
print(cm)
#print(classification_report(y_test, y_pred))
print("Accuracy:  " + str(accuracy_score(y_test4cm, y_pred)))
print("Recall:   " + str(recall_score(y_test4cm, y_pred, average='micro')))
print("Precision:        " + str(precision_score(y_test4cm, y_pred, average='micro')))

# run some examples
mylist = []

for i in range(0,10):
    x = random.randint(25000,28088)
    mylist.append(x)
    print(x)

with open(in_filename) as f:
    lines = f.readlines()

for v in mylist:
    otext = lines[v]
    print("Original: " + otext)
    owords = otext.split()
    seed_text=""
    num_words = len(otext.split())
    for i in range(0, num_words-2):
        seed_text += owords[i] + " "
    seed_text = seed_text.strip()
    print("Seed:     " + seed_text)
    print("Predict:  " + generate_seq(model, tokenizer, max_length-1, seed_text, 2))
    print('\n')

