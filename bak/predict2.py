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

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# convert text to integers for sequences
encoded = tokenizer.texts_to_sequences([doc])[0]
sequences = list()

# create sequences
for i in range(5, len(encoded)):
    sequence = encoded[i-5:i+1]
    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))

# take care of variable length sequences by adding padding
max_length = max([len(seq) for seq in sequences])


# load model
model = load_model('lstm.h5')

#evaluate model
mylist = []

for i in range(0,10):
    #x = random.randint(0,20000)
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



