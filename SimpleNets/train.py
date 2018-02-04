#-*- coding:utf8 -*-
'''
Created on May 3, 2017

@author: czm
'''

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense
import numpy as np
import json
import cPickle as pkl
from data import make_XY
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os

X_train,y_train,_ = make_XY('data/train.src','data/train.src.json','data/train.hter')

print 'X_train',X_train.shape
print 'y_train',y_train.shape

X_val,y_val,_ = make_XY('data/dev.src','data/train.src.json','data/dev.hter')
print 'X_val',X_val.shape
print 'y_val',y_val.shape

#隐单元迭代次数
timesteps = X_train.shape[1]
#输入数据的维数
data_dim = X_train.shape[2]


#excepted input data shape:(batch_size,timesteps,data_dim)
model = Sequential()
model.add(LSTM(200,return_sequences=True,
               input_shape=(timesteps,data_dim)))

model.add(LSTM(200,return_sequences=True))
model.add(LSTM(200))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mae',optimizer='rmsprop')

early_stopping = EarlyStopping(monitor='val_loss',patience=5)
checkpointer = ModelCheckpoint(filepath='model/simplenets.model',verbose=0, save_best_only=True)

#重新加载模型
if os.path.isfile('model/simplenets.model'):
    print 'reload model...'
    model = load_model('model/simplenets.model')

model.fit(X_train,y_train,shuffle=True,
          batch_size=64,epochs=10,validation_data=(X_val,y_val),
          callbacks=[early_stopping,checkpointer])





if __name__ == '__main__':
    pass