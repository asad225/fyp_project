# Load the input data
import pandas as pd
df = pd.read_excel('dataset.xlsx')

# Preprocess the data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Questions'])
sequences = tokenizer.texts_to_sequences(df['Questions'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Create one-hot encoded labels
labels = pd.get_dummies(df['Responses']).values

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define the model with an embedding layer
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,Dense, Dropout, Embedding
from keras.optimizers import SGD

model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    Bidirectional(LSTM(128)),
    Dense(600, activation='relu'),
    Dense(600, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

# Compile the model
# sgd = SGD(lr=0.001, decay=5e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)
model.save('model.h5')
