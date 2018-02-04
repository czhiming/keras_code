#coding:utf-8
'''
Created on Dec 13, 2017

@author: czm
'''
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np


model = Sequential()
model.add(Dense(200, input_dim=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adadelta',
              loss='mean_absolute_error')

X = np.random.random((1000, 100))
y = np.random.random((1000, 1))

model.fit(X, y, epochs=10, batch_size=32)



if __name__ == '__main__':
    pass