import pandas as pd
import numpy as np
from keras.models import  Model
from keras.layers import LSTM,Input,Bidirectional,Embedding,Activation,Dropout,Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,64,input_length=max_len)(inputs)
    layer = Bidirectional(LSTM(50))(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

if __name__ == "__main__":
    df = pd.read_csv('data/processed_data.csv')
    df = df[['text', 'y']]
    X_train, X_test, Y_train, Y_test = train_test_split(df['text'], df['y'], test_size=0.15)
    max_words = 20000
    max_len = 150
    tokenizer = Tokenizer(max_words)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
    model = RNN()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
              validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix, Y_test)
    print(accr)