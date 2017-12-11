from __future__ import print_function

import os
import sys
import csv
import re
import argparse
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.optimizers import SGD
from keras.models import Model, load_model


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'embedding')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'training')
MAX_SEQUENCE_LENGTH = 140
MAX_NB_WORDS = 1600
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
embeddings_index = {} # dictionary mapping word name to vector embedding

def predict(dir_path):
    model = load_model('cnn.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("Preprocess testing data.")
    count = 0
    if os.path.isdir(dir_path):
        pattern = re.compile(".*txt")
        for fname in sorted(os.listdir(dir_path)):
            if re.match(pattern, fname):
                output = open("result"+str(count), 'w')
                count += 1
                texts = []  # list of text samples, clear up for every file(each day's data)
                fpath = os.path.join(dir_path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.readline()
                while t:
                    texts.append(t)
                    t = f.readline()
                f.close()

                tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
                tokenizer.fit_on_texts(texts)
                sequences = tokenizer.texts_to_sequences(texts)
                data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
                class_prob = model.predict(data)
                class_category = []
                for prob in class_prob:
                    index = np.argmax(prob)+1
                    output.write("{0}\n".format(index))
                    class_category.append(index)
                print(class_category)

    else:
        print("{0} is not a directory.".format(dir_path))


def train_cnn():
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    print('Indexing word vectors.')

    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # prepare text samples and their labels
    print('Processing text dataset')

    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name) # path of each subdirectory of text: 2_mood/positive
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname) # 2_mood/positive/0
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.readline()
                    while t:
                        texts.append(t)

                        labels.append(label_id)
                        t = f.readline()
                    f.close()

    print('Found {0} texts, {1} groups'.format(len(texts), len(labels_index)))

    # vectorize the text samples into a 2D integer tensor
    print('Preprocessing data.')

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index)) # word_index is the list of word: index

    data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) # 199997 by 100
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=16,
              epochs=10,
              validation_data=(x_val, y_val))

    model.save('cnn_text.h5')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Command Error! Usage: python3 glove_cnn -o predict/train.")
        exit(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--operation", required=True,
    help="operation to do, either train or predict")
    args = vars(ap.parse_args())
    if args["operation"] == "train":
        train_cnn()
    if args["operation"] == "predict":
        predict("predict")

