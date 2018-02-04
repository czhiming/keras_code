#-*- coding:utf8 -*-
'''
Created on Mar 2, 2017

@author: czm
'''
import sys
import numpy

######################################
#  python make_result_file.py test.hter.result test.hter
######################################

if len(sys.argv) < 3:
    sys.exit(0)
    
our_file = open(sys.argv[1]) #预测值
gold_file = open(sys.argv[2]) #黄金标准

method_name = 'JXNU/word_embedding+RNN'

def get_our_gold(our_file,gold_file):
    our = []
    gold = []
    for i,lines in enumerate(our_file):
        our.append(float(lines.strip())) 
        gold.append(float(gold_file.readline().strip()))
    
    return our,gold

def get_index(i,y_hat):
    y_hat_ = sorted(enumerate(y_hat),key=lambda x:x[1])
    for key,value in enumerate(y_hat_):
        if i == value[0]:
            return key+1

#获得预测值和黄金标准
our,gold = get_our_gold(our_file, gold_file)

with open("predicted.csv", 'w') as _fout:
            for i, _y in enumerate(zip(our, gold)):
                print >> _fout,  "%s\t%d\t%f\t%d" % (method_name,i+1,_y[0],get_index(i,our))
with open('ref.csv','w') as _fout:
    for i, _y in enumerate(zip(our, gold)):
        print >> _fout,  "%s\t%d\t%f\t%d" % ('SHEFF/QuEst',i+1,_y[1],get_index(i,gold))

our_file.close()
gold_file.close()



if __name__ == '__main__':
    pass
