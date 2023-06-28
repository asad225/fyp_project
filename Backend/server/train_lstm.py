import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet as wn
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import random


# Function to perform synonym replacement
def replace_synonyms(sentence):
    words = sentence.split()
    augmented_sentence = []
    for word in words:
        synsets = wn.synsets(word)
        if synsets:
            synonyms = [syn.lemmas()[0].name() for syn in synsets]
            synonym = random.choice(synonyms)
            augmented_sentence.append(synonym)
        else:
            augmented_sentence.append(word)
    return ' '.join(augmented_sentence)

df = pd.read_excel('BankFAQs.xlsx')

# Apply data augmentation to create additional training examples
augmented_questions = []
augmented_labels = []

for question, label in zip(df['Questions'], df['Responses']):
    augmented_questions.append(question)
    augmented_labels.append(label)
    
    paraphrase = replace_synonyms(question)
    augmented_questions.append(paraphrase)
    augmented_labels.append(label)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(augmented_questions)
sequences = tokenizer.texts_to_sequences(augmented_questions)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Create one-hot encoded labels
labels = pd.get_dummies(augmented_labels).values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define and compile the model
# model = Sequential([
#     Embedding(input_dim=10000, output_dim=64, input_length=100),
#     Bidirectional(LSTM(256, return_sequences=True)),
#     LSTM(128),
#     Dense(1024, activation='relu'),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(labels.shape[1], activation='softmax')
# ])


# model = Sequential([
#     Embedding(input_dim=10000, output_dim=64, input_length=100),
#     Bidirectional(LSTM(128)),
#     Dense(600, activation='gelu'),
#     Dense(600, activation='gelu'),
#     Dropout(0.5),
#     Dense(labels.shape[1], activation='softmax')
# ])

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    Bidirectional(LSTM(256, return_sequences=True)),
    Bidirectional(LSTM(128)),
    Dense(512, activation='gelu'),
    Dropout(0.5),
    Dense(256, activation='gelu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)

model.save('lstm_BankFAQs.h5')
