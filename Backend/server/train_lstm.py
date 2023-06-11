
# Import the required libraries
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from transformers import pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import SGD
from transformers import T5Tokenizer, T5ForConditionalGeneration


df = pd.read_excel('BankFAQs.xlsx')
# Load the T5 model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate paraphrases using back-translation
def generate_paraphrase(text):
    # Translate to a different language
    translated = model.generate(input_ids=tokenizer.encode(text, return_tensors='pt'), 
                               max_length=100, 
                               num_return_sequences=1,
                               early_stopping=True,
                               decoder_start_token_id=model.config.pad_token_id)
    
    # Translate back to the original language
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    back_translated = model.generate(input_ids=tokenizer.encode(translated_text, return_tensors='pt'), 
                                     max_length=100, 
                                     num_return_sequences=1,
                                     early_stopping=True,
                                     decoder_start_token_id=model.config.pad_token_id)
    
    # Return the back-translated paraphrase
    paraphrase = tokenizer.decode(back_translated[0], skip_special_tokens=True)
    return paraphrase

# Apply data augmentation to create additional training examples
augmented_questions = []
augmented_labels = []

for question, label in zip(df['Questions'], df['Responses']):
    augmented_questions.append(question)
    augmented_labels.append(label)
    
    paraphrase = generate_paraphrase(question)
    augmented_questions.append(paraphrase)
    augmented_labels.append(label)

# Tokenize and pad sequences
tokenizer.fit_on_texts(augmented_questions)
sequences = tokenizer.texts_to_sequences(augmented_questions)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Create one-hot encoded labels
labels = pd.get_dummies(augmented_labels).values

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Define and compile the model (using the architecture from the previous example)
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    Bidirectional(LSTM(256, return_sequences=True)),
    LSTM(128),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)

model.save('lstm_bankfaq.h5')
