import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
# Read the data
df = pd.read_excel('dataset.xlsx')

# Preprocess the data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Questions'])
X = tokenizer.texts_to_sequences(df['Questions'])
X = pad_sequences(X, maxlen=100, padding='post', truncating='post')
label_encoder = LabelEncoder()
label_encoder.fit(df['Responses'])
y = label_encoder.transform(df['Responses'])

# Split the data into training and validation sets
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
split = int(0.8 * X.shape[0])
X_train = X[:split]
X_val = X[split:]
y_train = y[:split]
y_val = y[split:]

# Define the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    Bidirectional(LSTM(256)),
    Dense(600, activation='gelu'),
    Dense(600, activation='gelu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=50,
          epochs=150)

# Save the model
model.save('model.h5')