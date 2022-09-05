import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer

intetns = json.loads(open("intetns.json").read())

words = []
classes = []
documents = []
ignoredLatters = ["?", "!", ",", ".", "'"]

for intent in intetns["intents"]:
    for pattern in intetns["patterns"]:
        # Tokenize splits every word
        wordList = nltk.word_tokenize(pattern)
        words.append(wordList)
        # Current wordlist is linked to the intents tag
        documents.append((wordList, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
print(documents)
