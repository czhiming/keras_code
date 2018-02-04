#-*- coding:utf8 -*-
'''
Created on Apr 25, 2017

@author: czm
'''
from keras.models import Sequential
from keras.layers import LSTM,Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

#excepted input data shape: (batch_size,timesteps,data_dim)
model = Sequential()
model.add(LSTM(32,return_sequences=True,
               input_shape=(timesteps,data_dim)))
model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Generate dummpy training data
x_train = np.random.random((1000,timesteps,data_dim))
y_train = np.random.random((1000,num_classes))

#Generate dummpy validation data
x_val = np.random.random((100,timesteps,data_dim))
y_val = np.random.random((100,num_classes))


model.fit(x_train,y_train,
          batch_size=64,epochs=5,
          validation_data=(x_val,y_val))








if __name__ == '__main__':
    pass