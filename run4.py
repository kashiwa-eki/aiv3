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
#  and to: https://adventuresinmachinelearning.com/keras-lstm-tutorial/
#


from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout, TimeDistributed, Activation
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
import numpy as np
import random
import sys
import argparse
import pickle

data_path = "data"
source_file = data_path + "/review.dat"
train_file = data_path + "/train.txt"
test_file = data_path + "/test.txt"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path


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
        
# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
        in_text = seed_text
        out_text = ''
        # generate a fixed number of words (2 for our example)
        for _ in range(n_words):
                # encode the text as integer
                encoded = tokenizer.texts_to_sequences([in_text])[0]
                # pre-pad sequences to a fixed length
                encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
                # predict probabilities for each word
                yhat = model.predict_classes(encoded, verbose=0)

                for word, index in tokenizer.word_index.items():
                        if index == yhat:
                                out_word = word
                                break
                # append to input
                in_text += ' ' + out_word
                out_text += ' ' + out_word
        return out_text.strip()


if args.run_opt == 1:

    # read source and shuffle lines
    with open(source_file, "r") as f:
        data = f.read().split('\n')

    random.shuffle(data)

    print(data[:10])

    # create training and test datasets
    count = len(data)
    numrows = round(count * 0.8)
    print(numrows)
    
    train_doc = ''
    test_doc = ''

    cnt = 1
    for line in data:
        #print(line)
        if cnt > numrows:
            test_doc += line + '\n'
        else:
            train_doc += line + '\n'
        cnt += 1

    with open(train_file, 'w') as f:
        f.write(train_doc)
    with open(test_file, 'w') as f:
        f.write(test_doc)


    # prepare the tokenizer on the source text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([train_doc])

    # save tokenizer for making predictions later
    #with open('data/tokenizer.pickle', 'wb') as handle:
    #    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # determine the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    encoded = tokenizer.texts_to_sequences([train_doc])[0]

    # create line-based sequences
    sequences = list()
    for line in train_doc.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        print(encoded)

        if len(encoded) > 7:
            startnum = len(encoded) - 7
        else:
            startnum = 0
        for i in range(startnum, len(encoded)-2):
            sequence = encoded[i:]
            print(sequence)
            sequences.append(sequence)
    #sys.exit()

    print('Total Sequences: %d' % len(sequences))
    
    # get max length of entire sentence for padding
    max_length = 0
    for line in train_doc.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        #print(encoded)
        if len(encoded) > max_length:
            max_length = len(encoded)

    print('Max Length: %d' % max_length)
    max_length = 7
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)

    # assign input/output elements for model
    sequences = array(sequences)
    X_train, y_train = sequences[:,:-2],sequences[:,-2]

    # convert y values to one hot encoding
    y_train = to_categorical(y_train, num_classes=vocab_size)

    # create model with single hidden LSTM layer with 512 memory units
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_length-1))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # compile model and evaluate using accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model using training data. validate using 10% of training data which is not trained on. 5 epochs and batch size of 128.
    model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=10, verbose=2)

    # save model
    model.save(data_path + '/lstm_model.h5')

elif args.run_opt == 2:
    # loading
    with open(data_path + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load the model
    model = load_model(data_path + "/lstm_model.h5")
    
    y_true = []
    y_pred = []
    num_right = 0
    num_wrong = 0
    word1_right = 0
    word1_wrong = 0
    word2_right = 0
    word2_wrong = 0
    total = 0
    max_length = 14

    # evaluate model using untrained test data
    with open(test_file, 'r') as f:
        for line in f:
            otext = line
            print("Original: " + otext)
            # split into words
            owords = otext.split()
            seed_text= ''
            # get last two words of sentence
            word1_true = owords[len(owords)-2]
            word2_true = owords[len(owords)-1]
            end_true = word1_true + ' ' + word2_true
            print("Actual:   " + end_true)
            # append to y_true array
            y_true.append(end_true)
            # build seed text (sentence minus last two words)
            if len(owords) > 4:
                startnum = len(owords) - 4
            else:
                startnum = 0
            for i in range(startnum, len(owords)-2):
                seed_text += owords[i] + " "
            seed_text = seed_text.strip()
            #print("Seed:     " + seed_text)
            # get predicted last two words
            end_pred = generate_seq(model, tokenizer, max_length-1, seed_text, 2)
            print("Predict:  " + end_pred)
            # append to y_pred array
            y_pred.append(end_pred)
            # get each predicted word
            outwords = end_pred.split()
            word1_pred = outwords[0]
            word2_pred = outwords[1]
            print('-----------------\n')
            if end_true == end_pred:
                num_right += 1
            else:
                num_wrong += 1
            total += 1
            if word1_true == word1_pred:
                word1_right += 1
            else:
                word1_wrong += 1
            if word2_true == word2_pred:
                word2_right += 1
            else:
                word2_wrong += 1

    # print all metrics
    print("Total Sequences:    " + str(total))
    print("--------------")
    print("Last Two Correct:   " + str(num_right))
    print("Last Two Incorrect: " + str(num_wrong))
    print("Last Two Accuracy:  " + str(num_right / total * 100.0))
    print("--------------")
    print("Word 1 Correct:     " + str(word1_right))
    print("Word 1 Incorrect:   " + str(word1_wrong))
    print("Word 1 Accuracy:    " + str(word1_right / total * 100.0))
    print("--------------")
    print("Word 2 Correct:     " + str(word2_right))
    print("Word 2 Incorrect:   " + str(word2_wrong))
    print("Word 2 Accuracy:    " + str(word2_right / total * 100.0))
    print("--------------")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    #print(classification_report(y_true, y_pred))
    print("Accuracy:  " + str(accuracy_score(y_true, y_pred)))
    #print("Recall:   " + str(recall_score(y_true, y_pred, average='micro')))
    #print("Precision:        " + str(precision_score(y_true, y_pred, average='micro')))


