#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 08:34:00 2018

@author: flyaway
"""

#train and test ffm


import os
import numpy as np
import pandas as pd
from sklearn import  preprocessing
import gc
from tqdm import tqdm
from itertools import izip
def print_and_exec(cmd):
    os.system(cmd)
    
    
def train_model(opts,tr_file,val,va_file=None):
    print "  \n Training..."
    if val == True:
        opts += " -p" + tr_file
    print_and_exec("/home/flyaway/libffm/ffm-train %s %s ./ffm_model/model" % (opts, tr_file))

def predict(file_path,name = "train"):
    print "  Predicting..."
    print_and_exec("/home/flyaway/libffm/ffm-predict %s ./ffm_model/model %s.preds" % (te_file,name))

if __name__ == "__main__":
    online = True
    opt = " -l 0.00002 -k 4 -t 20 -r 0.2 -s 16 "
    if online == False:
        tr_file = "./offline/tr.txt"
        va_file = "./oflline/tr.txt"
        #train_model(opt,tr_file,val=True,va_file)
    else:
        tr_file = "./online/tr.txt"
        #train_model(opt,val=False)
        
        test = pd.read_csv('./round1_ijcai_18_test_a_20180301.txt', sep=' ')
        te_file = "./online/te.txt"
        #predict(te_file,name = "test")
        test['predicted_score'] = np.loadtxt('./test.preds')
        test[['instance_id', 'predicted_score']].to_csv('ffm_test_baseline.csv', index=False,sep=' ')
        """
        train = pd.read_csv('./round1_ijcai_18_train_a_20180301.txt', sep=' ')
        tr_file = "./online/tr.txt"
        predict(tr_file,name = "train")
        train['predicted_score'] = np.loadtxt('./train.preds')
        train[['instance_id', 'predicted_score']].to_csv('ffm_train_baseline.csv', index=False,sep=' ')
        """