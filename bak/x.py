import random
import sys


data_path = "datax"
source_file = data_path + "/review.dat"
train_file = data_path + "/train.txt"
test_file = data_path + "/test.txt"



def load_doc(filename):
        file = open(filename, 'r')
        text = file.read()
        #random.shuffle(text)
        file.close()
        return text


doc = load_doc("data/review.dat")

with open(source_file, "r") as f:
    data = f.read().split('\n')

random.shuffle(data)

print(data[:10])
print('----------')
print(doc[:10])


count = len(data)
numrows = round(count * 0.8)

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




sys.exit()
from keras.preprocessing.text import Tokenizer

# prepare the tokenizer on the source text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([train_doc])

# create line-based sequences
sequences = list()
for line in data: #.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    print(encoded)

    for i in range(0, len(encoded)-1):
        sequence = encoded[i:]
        #print(sequence)


