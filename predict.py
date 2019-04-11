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

# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([doc])

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# encode 2 words -> 1
#encoded = tokenizer.texts_to_sequences([doc])[0]
#sequences = list()
#for i in range(6, len(encoded)):
#        sequence = encoded[i-6:i+1]
#        sequences.append(sequence)
#        print(sequence)
#print('Total Sequences: %d' % len(sequences))

# create line-based sequences
sequences = list()
for line in doc.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
#       sequence = encoded[:len(encoded)-1]
#       sequences.append(sequence)
#       print(sequence)
#       sequence = encoded[:]
#       sequence = encoded[:]
#       sequences.append(sequence)
#       print(sequence)
#       sequence = encoded[:]
        for i in range(-1, len(encoded)-3):
                sequence = encoded[i+1:]
                print(sequence)
                sequences.append(sequence)
        for i in range(-1, len(encoded)-4):
                sequence = encoded[i+1:len(encoded)-1]
                print(sequence)
                sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))

# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

#print(sequences)

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# load model
model = load_model('lstm6.h5')

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



