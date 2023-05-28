import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
# # Load the model
model = load_model('model.h5')
label_encoder = LabelEncoder()

df = pd.read_excel('dataset.xlsx')
label_encoder.fit(df['Responses'])
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Questions'])


if __name__ == '__main__':
    correct = 0
    for question,orig_answer in zip(df['Questions'],df['Responses']):
        sequence = tokenizer.texts_to_sequences([question])
        padded_sequence = pad_sequences(
            sequence, maxlen=100, padding='post', truncating='post')
        prediction = model.predict(padded_sequence)

        # Get the predicted answer
        answer = label_encoder.inverse_transform([np.argmax(prediction)])
        # print("Question:",question,'\n\n')
        # print("Response:",answer[0],'\n\n')
        # print("Expected Response:",orig_answer,'\n\n')
        if answer[0] == orig_answer:
            correct = correct + 1
        # time.sleep(5)

    total = df.shape[0]
    print("Accuracy",round((correct/total)*100,2),'%')
