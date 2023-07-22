import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
<<<<<<< Updated upstream
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
=======
from tensorflow.keras.optimizers import SGD
import nlpaug.augmenter.word as naw

# Load the input data
df = pd.read_excel('internet_dataset.xlsx')
>>>>>>> Stashed changes

# Augment the data with paraphrases
aug = naw.SynonymAug(aug_src='wordnet')
df_augmented = pd.DataFrame(columns=['Questions', 'Responses'])
for index, row in df.iterrows():
    question = row['Questions']
    paraphrases = [aug.augment(question) for _ in range(2)]  # Generate 2 paraphrases
    for paraphrase in paraphrases:
        df_augmented = pd.concat([df_augmented, pd.DataFrame({'Questions': [paraphrase], 'Responses': [row['Responses']]})], ignore_index=True)
df = pd.concat([df, df_augmented])

# Preprocess the data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Questions'])
sequences = tokenizer.texts_to_sequences(df['Questions'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Create one-hot encoded labels
labels = pd.get_dummies(df['Responses']).values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

<<<<<<< Updated upstream
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
=======
# Define the model with an embedding layer
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    Bidirectional(LSTM(128)),
    Dense(600, activation='relu'),
    Dense(600, activation='relu'),
>>>>>>> Stashed changes
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

<<<<<<< Updated upstream

=======
# Compile the model
>>>>>>> Stashed changes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)

<<<<<<< Updated upstream
model.save('lstm_BankFAQs.h5')
=======
# Save the model
model.save('para_internet_dataset.h5')
>>>>>>> Stashed changes
