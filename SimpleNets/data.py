#-*- coding:utf8 -*-
'''
Created on May 3, 2017

@author: czm
'''
import numpy as np
import json
import cPickle as pkl
import os
import sys
from collections import OrderedDict

def ortho_weight(ndim):
    W = np.random.randn(ndim,ndim)
    u,s,v = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None,scale=0.01,ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale*np.random.randn(nin,nout)
    return W.astype('float32')

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)

def make_ngram(lines,ngrams):
    ngrams_list = []
    word_list = lines.strip().split()
    for i in range(len(word_list)-ngrams+1):
        ngrams_list.append(' '.join(word_list[i:i+ngrams]))
    return ngrams_list

def make_XY(file_name,file_json,file_hter=None,ngrams=2,dim_words=200):
    if os.path.isfile(file_name) == False:
        print 'This file:',file_name,'is not exists!!!'
        sys.exit(0)
    ngram_dict = OrderedDict()
    
    if file_hter is not None:
        with open(file_hter) as fp:
            HTER = []
            for lines in fp:
                hter = float(lines.strip())
                HTER.append(hter)
    
    with open(file_name) as fp: 

        for i,lines in enumerate(fp):
            ngram_list = make_ngram(lines,ngrams)
            for ngram in ngram_list:
                if ngram not in ngram_dict:
                    ngram_dict[ngram] = [None]*2
                    ngram_dict[ngram][0] = 0
                    ngram_dict[ngram][1] = 0.
                ngram_dict[ngram][0] += 1
                if file_hter is not None:
                    ngram_dict[ngram][1] += HTER[i]
            
    word_dict = load_dict(file_json)
    numbers = len(ngram_dict)
    timesteps = dim_words
    data_dim = 2
    
    #获得词向量
    #Wemb = norm_weight(len(word_dict),dim_words)
    with open('data/train.src.json.emb') as fp:
        Wemb = pkl.load(fp)
    
    X_train = []
    y_train = []
    ngram_list = []
    
    for ngram in ngram_dict:
        ngram_matrix = []
        words = ngram.split()
        #判断词是否在字典里
        flag = 1
        for word in words:
            if word not in word_dict:
                flag = 0
                break
        if flag == 1:
            for word in words:
                ngram_matrix.append(list(Wemb[word_dict[word]]))
                
            ngram_list.append(ngram)
            X_train.append(ngram_matrix)
            if file_hter is not None:
                y_train.append(ngram_dict[ngram][1]/ngram_dict[ngram][0])
    if file_hter is not None:
        X_train = np.array(X_train)
        shapes = X_train.shape
        X_train = X_train.reshape([shapes[0],shapes[2],shapes[1]])
        y_train = np.array(y_train)
        
        return X_train,y_train,ngram_list
    else:
        X_train = np.array(X_train)
        shapes = X_train.shape
        X_train = X_train.reshape([shapes[0],shapes[2],shapes[1]])
        
        return X_train,_,ngram_list






if __name__ == '__main__':
    pass