#!/bin/sh

device=gpu

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python fnn_qe.py


