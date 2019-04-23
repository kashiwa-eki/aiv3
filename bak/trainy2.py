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
from keras.layers import Dropout
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

data_path = "/home/kashiwa/aiv3/data"

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
        
# return words from sequence
def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

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
    # source data
    in_filename = "data/review.dat"
    doc = load_doc(in_filename)
    count = len(open(in_filename).readlines())
    numrows = round(count * 0.8)

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

    with open('data/train.txt', 'w') as f:
        f.write(train_doc)
    with open('data/test.txt', 'w') as f:
        f.write(test_doc)


    # prepare the tokenizer on the source text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([doc])

    with open('data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # determine the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # convert text to integers for sequences
    #encoded = tokenizer.texts_to_sequences([train_doc])[0]


    # create line-based sequences
    sequences = list()
    for line in train_doc.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        #for i in range(0, len(encoded)):
        #    for word, index in tokenizer.word_index.items():
        #        if index == encoded[i]:
        #            print(word)
        #print(len(encoded))
        print(encoded)

        if len(encoded) > 8:
            startnum = len(encoded) - 8
        else:
            startnum = 0
        for i in range(startnum, len(encoded)-1):
            sequence = encoded[i:]
            print(sequence)
            sequences.append(sequence)

        if len(encoded) > 9:
            startnum = len(encoded) - 9
        else:
            startnum = 0
        for i in range(startnum, len(encoded)-2):
            sequence = encoded[i:len(encoded)-1]
            print(sequence)
            sequences.append(sequence)

    print('Total Sequences: %d' % len(sequences))
    #sys.exit() 
    max_length = 0
    for line in doc.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        #print(encoded)
        if len(encoded) > max_length:
            max_length = len(encoded)

    print('Max Length: %d' % max_length)

    # take care of variable length sequences by adding padding
    #train_length = max([len(seq) for seq in sequences])
    #test_length = max([len(seq) for seq in test_sequences])
    #max_length = max(train_length, test_length)
    #max_length = train_length
    #print(max_length)
    #sys.exit()

    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    #test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='pre')
    print('Max Sequence Length: %d' % max_length)

    #assign input/output elements for model
    sequences = array(sequences)
    X_train, y_train = sequences[:,:-1],sequences[:,-1]
    #test_sequences = array(test_sequences)
    #X_test, y_test = test_sequences[:,:-1],test_sequences[:,-1]

    print(sequences[0])
    print(X_train[0])
    print(y_train[0])
#print(X_test[0])
#print(y_test[0])

#print(len(sequences))
#print(len(X))
#print(len(y))
#numrows = round(len(X) * 0.8)
#print(numrows)
#X_train = X[:numrows]
#y_train = y[:numrows]
#X_test = X[numrows:]
#y_test = y[numrows:]

#print(X_train[numrows-1])
#print(y_train[numrows-1])
#print(sequences[numrows-1])
#print(X_test[0])
#print(y_test[0])
#print(sequences[numrows])
#print(list(map(sequence_to_text, X_train[numrows-1])))
#print(list(map(sequence_to_text, X_test[0])))
#print(" ".join([reverse_word_map[x] for x in X_test[:10]]))

#sys.exit()

# split data into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
    y_train = to_categorical(y_train, num_classes=vocab_size)
#y_test4cm = y_test
#y_test = to_categorical(y_test, num_classes=vocab_size)


    # create model with single hidden LSTM layer with 128 memory units
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_length-1))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # compile model and evaluate using accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit model using training data. 30 epochs and batch size of 32.
    model.fit(X_train, y_train, validation_split=0.1, batch_size=128, epochs=5, verbose=2)

    # save model
    model.save('data/lstm_model.h5')

elif args.run_opt == 2:
    # loading
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # evaluate model
    #score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
    #print('Test accuracy: ' + str(score[1]))

    # predict the test set results
    #y_pred = model.predict_classes(X_test, batch_size=32, verbose=1)
    #print(y_pred)
    #print(y_test4cm)

    # create a confusion matrix
    #cm = confusion_matrix(y_test4cm, y_pred)
    #print('Confusion Matrix')
    #print(cm)
    #print(classification_report(y_test, y_pred))
    #print("Accuracy:  " + str(accuracy_score(y_test4cm, y_pred)))
    #print("Recall:   " + str(recall_score(y_test4cm, y_pred, average='micro')))
    #print("Precision:        " + str(precision_score(y_test4cm, y_pred, average='micro')))

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

    # evaluate
    with open('data/test.txt') as f:
        for line in f:
            otext = line
            print("Original: " + otext)
            owords = otext.split()
            seed_text= ''
            end_text = ''
            end_text = owords[len(owords)-2] + ' ' + owords[len(owords)-1]
            word1_true = owords[len(owords)-2]
            word2_true = owords[len(owords)-1]
            end_true = word1_true + ' ' + word2_true
            print("Actual:   " + end_true)
            y_true.append(end_true)
            for i in range(0, len(owords)-2):
                seed_text += owords[i] + " "
            seed_text = seed_text.strip()
            #print("Seed:     " + seed_text)
            end_pred = generate_seq(model, tokenizer, max_length-1, seed_text, 2)
            print("Predict:  " + end_pred)
            y_pred.append(end_pred)
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

    print(num_right)
    print(num_wrong)
    print(total)
    print(num_right / total * 100.0)
    print(word1_right)
    print(word1_wrong)
    print(word1_right / total * 100.0)
    print(word2_right)
    print(word2_wrong)
    print(word2_right / total * 100.0)
    print(y_true[:10])
    print(y_pred[:10])

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    #print(classification_report(y_true, y_pred))
    print("Accuracy:  " + str(accuracy_score(y_true, y_pred)))
    print("Recall:   " + str(recall_score(y_true, y_pred, average='micro')))
    print("Precision:        " + str(precision_score(y_true, y_pred, average='micro')))


