#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import wandb
from wandb.keras import WandbCallback
import re


run = wandb.init(project="sttextgen")
config = run.config
config.batch_size = 768
config.file = r"C:\Datasets\Text\Stranger Things\Season 1\1.txt"
config.maxlen = 128 #Len of sliding window
config.step = 3 #The amount of characters per step
config.epochs = 250
config.charsGen = 500
config.rememberChars = 200


strDelChars = '[01234567890->:,]'

text = io.open(config.file, encoding='utf-8').read()
text = re.sub(strDelChars, '', text)
text = text.replace("\n\n", "\n").replace("-", "").replace("\n ", "").replace("\n ", "")

with io.open(r"C:\Coding\Python\ML\Text\normalizedText.txt", "w", encoding='utf-8') as newFile:
    newFile.write(text)
chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build a sequence for every <config.step>-th character in the text

sentences = []
next_chars = []
for i in range(0, len(text) - config.maxlen, config.step):
    sentences.append(text[i: i + config.maxlen])
    next_chars.append(text[i + config.maxlen])

# build up one-hot encoded input x and output y where x is a character
# in the text y is the next character in the text

x = np.zeros((len(sentences), config.maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#Creating Model
model = keras.models.load_model(r'C:\Coding\Python\ML\Text\Models\STS1_CharGen.gen')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def printText():
    start_index = random.randint(0, len(text) - config.maxlen - 1)

    for diversity in [1.2]:
        generated = ''
        sentence = text[start_index: start_index + config.maxlen]
        generated += f"{sentence}"
        print('"\n\n\nGenerated:\nDiversity: ' + str(diversity) +"\n\n")
        sys.stdout.write(generated)

        for i in range(config.charsGen):
            x_pred = np.zeros((1, config.maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

printText()