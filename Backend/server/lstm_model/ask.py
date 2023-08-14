import numpy as np
import pandas as pd
import time
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
# # Load the model
model = load_model('internet_dataset_train.h5')
label_encoder = LabelEncoder()

df = pd.read_excel('internet_dataset.xlsx')
label_encoder.fit(df['Responses'])
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Questions'])



def get_bot_response(question):
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(
    sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    answer = label_encoder.inverse_transform([np.argmax(prediction)])
    return answer[0]

     

    
if __name__ == '__main__':
    # while True:
    #     input_data = input('You: ')
    #     response = get_bot_response(input_data)
    #     print('bot: ' + response)

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
