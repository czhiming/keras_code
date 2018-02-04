#-*- coding:utf8 -*-
'''
Created on May 3, 2017

@author: czm
'''
from keras.models import load_model
import os
from data import make_XY

if os.path.isfile('model/simplenets.model'):
    model = load_model('model/simplenets.model')

X_val,y_val,ngram_list = make_XY('data/dev.src','data/train.src.json','data/dev.hter')
print 'X_val',X_val.shape
print 'y_val',y_val.shape


result = model.predict(X_val)

print result.shape

for i in range(result.shape[0]):
    try:
        print ngram_list[i],result[i]
    except:
        print 'error !!!'








if __name__ == '__main__':
    pass