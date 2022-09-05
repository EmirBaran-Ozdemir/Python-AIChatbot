import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignoredLatters = ["?", "!", ",", ".", "'"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize splits every word
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        # Current wordlist is linked to the intents tag
        documents.append((wordList, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Splitted words listed and sorted also ignored words excluded
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoredLatters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Machine Learning Part
# Converting words to numerical values to feed into neural network
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    # If a word is in wordPatterns fill bag with 1 else 0
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    # Copying list not typecasting
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append([bag, outputRow])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Building neural network application
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(
    np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1
)
model.save("chatbotModel.h5", hist)
print("done")
