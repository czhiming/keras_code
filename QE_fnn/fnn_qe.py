#coding:utf8

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adadelta
from keras.callbacks import EarlyStopping,ModelCheckpoint 
from keras import backend as K
from keras.initializers import random_uniform,glorot_uniform
from keras.regularizers import l1,l2
from keras.models import load_model

import sys
import os
import numpy as np
from sklearn.preprocessing import scale,StandardScaler
from compiler.syntax import check

#设置随机种子
np.random.seed(1234) 

def read_feature(file):
    features = []
    with open(file) as fp:
        for lines in fp:
            lines = lines.strip().split('\t')
            features.append(lines)
    return np.array(features)

def read_hter(file):
    hter = []
    with open(file) as fp:
        for lines in fp:
            lines = lines.strip()
            hter.append(float(lines))
    return np.array(hter)

def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

print 'load data...'
#加载数据
X_train = read_feature('wmt17_en_de/train.features')
y_train = read_hter('wmt17_en_de/train.hter')
X_test = read_feature('wmt17_en_de/dev.features')
y_test = read_feature('wmt17_en_de/dev.hter')

#数据标准化处理
try:
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
except:
    pass

print 'Build Model...'
#-----创建模型------
model = Sequential()
#隐层
model.add(Dense(100,input_dim=17,activation='relu'))
model.add(Dropout(0.2))
#隐层
# model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.2))
#隐层
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

if os.path.isfile('model/fnn_qe.hdf5'):
    model = load_model('model/fnn_qe.hdf5')
    
#建立优化器
adadelta = Adadelta(lr=0.001)#设定学习速度，衰减量等
#配置学习过程
model.compile(loss=mean_absolute_error,optimizer=adadelta)

#model.summary()
#sys.exit(0)
#print model.layers

#开始训练
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpointer = ModelCheckpoint(filepath="model/fnn_qe.hdf5", verbose=0, save_best_only=True)  
print 'Optimizer...'
#每次打乱顺序shuffle=True，epochs大小，batch_size大小，验证集
hist = model.fit(X_train,y_train,epochs=1000,batch_size=64,shuffle=True,\
          validation_data=(X_test,y_test),callbacks=[early_stopping,checkpointer],\
          verbose=1)
#model.save_weights('fnn_qe_weights.h5')
#输出历史loss值
#print hist.history

result = model.predict(X_test)
#print result.shape

print 'save result...'
with open('dev.predicted.hter','w') as fp:
    for hter in result:
        fp.writelines(str(hter[0])+"\n")























