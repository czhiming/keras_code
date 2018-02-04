#-*- coding:utf8 -*-
'''
Created on Apr 25, 2017

@author: czm
'''
from keras.models import Sequential
from keras.layers import Embedding
import numpy as np

model = Sequential()
model.add(Embedding(1000, 5))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

#input_array = np.random.randint(1000, size=(32, 10))

input_array = np.array([[2,4,88,87,999,1],
                        [1,0,55,33,22,0],
                        [2,3,0,111,8,0]])

model.compile(optimizer='rmsprop',loss='mse')


output_array = model.predict(input_array)

print output_array.shape
print output_array

#assert output_array.shape == (32, 10, 64)





















if __name__ == '__main__':
    pass