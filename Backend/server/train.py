import random
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import os
import time

ignore_words = ["?", "!"]

def tokenize_words_and_prepare_docs_classes(intents):
    words = []
    classes = []
    documents = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:

            # take each word and tokenize it
            w = nltk.word_tokenize(pattern)
            words.extend(w)

            # adding documents
            documents.append((w, intent["tag"]))

            # adding classes to our class list
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    return words,classes,documents

def lemmatization(words):
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    return words

def prepare_training_data(classes,documents,words):
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        # initializing bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        # create our bag of words array with 1, if word match found in current pattern
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    print("Training data created")
    return train_x,train_y

# Actual training
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax
def create_model(train_x,train_y,epochs=200,batch_size=5):
    
    model = Sequential()
    model.add(Dense(120, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(train_y[0]), activation="softmax"))
    model.summary()

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # for choosing an optimal number of training epochs to avoid underfitting or overfitting use an early stopping callback to keras
    # based on either accuracy or loos monitoring. If the loss is being monitored, training comes to halt when there is an 
    # increment observed in loss values. Or, If accuracy is being monitored, training comes to halt when there is decrement observed in accuracy values.

    # from keras import callbacks 
    # earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
    # callbacks =[earlystopping]

    # fitting and saving the model
    history = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("model.h5", history)
    print("model created")

if __name__ == '__main__':

    os.system("python preprocess.py")
    time.sleep(3)

    nltk.download('omw-1.4')
    nltk.download("punkt")
    nltk.download("wordnet")

    lemmatizer = WordNetLemmatizer()

    dataset = open("intents.json").read()
    intents = json.loads(dataset)

    words,classes,documents = tokenize_words_and_prepare_docs_classes(intents)

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

    train_x,train_y = prepare_training_data(classes,documents,words)

    create_model(train_x,train_y,200,5)








