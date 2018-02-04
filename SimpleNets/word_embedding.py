#-*- coding:utf8 -*-
'''
Created on Mar 1, 2017

@author: czm
'''
import gensim
import numpy
import argparse
import six.moves.cPickle as pickle
import json


parser = argparse.ArgumentParser()

parser.add_argument('-dim',type=str,required=True,help='the dimension of the word embedding')
parser.add_argument('-m',type=str,required=True,help='the model use for training word embedding')
parser.add_argument('-i',type=str,required=True,help='the input file name')
parser.add_argument('-o',type=str,required=True,help='the output file name')

args = parser.parse_args()


print 'model:',args.m

#model = gensim.models.Word2Vec.load_word2vec_format(args.m,binary=True)
model = gensim.models.KeyedVectors.load_word2vec_format(args.m,binary=True)

print 'file:',args.i
with open(args.i) as fi:
    word_dict = json.load(fi)


with open(args.o,'w') as fo:
    emb = []
    for word in word_dict:
        try:
            emb.append(list(model[word])) 
        except:
            emb.append(list(numpy.zeros(int(args.dim))))
    pickle.dump(emb,fo)


if __name__ == '__main__':
    pass
