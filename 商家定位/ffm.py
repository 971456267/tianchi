#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:17:19 2017

@author: flyaway
"""

#train ffm
import os
import numpy as np
import pandas as pd
from sklearn import  preprocessing
import gc
from tqdm import tqdm
from itertools import izip
path = "../"
df=pd.read_csv(path+u'训练数据-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'训练数据-ccf_first_round_shop_info.csv')
test=pd.read_csv(path+u'AB榜测试集-evaluation_public.csv')
#train_res = res = np.memmap("train_ffm.npy", dtype='float32', mode='w+', shape=(df.shape[0], 220))
test_res = np.memmap("test_ffm.npy", dtype='float32', mode='w+', shape=(test.shape[0], 220))

del df,shop,test
gc.collect()

#train_start = 0
test_start = 0
#train_end = 0
test_end = 0
#options:
#-l <lambda>: set regularization parameter (default 0.00002)
#-k <factor>: set number of latent factors (default 4)
#-t <iteration>: set number of iterations (default 15)
#-r <eta>: set learning rate (default 0.2)
#-s <nr_threads>: set number of threads (default 1)
#-p <path>: set path to the validation set
#--quiet: quiet model (no output)
#--no-norm: disable instance-wise normalization
#--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)


def print_and_exec(cmd):
    #print cmd
    os.system(cmd)
    
    
def train(opts,train_file,val = True):

    print "  \n Training..."
    
    if val == True:
        opts += " -p " + train_file

    print_and_exec("/home/flyaway/libffm/ffm-train %s %s ../ffm_model/ffm.model" % (opts, train_file))



def predict(test_file):
    print "  Predicting..."
    print_and_exec("/home/flyaway/libffm/ffm-predict %s ../ffm_model/ffm.model ./ffm/ffm.preds" % (test_file))

if __name__ == '__main__':
   
    result=pd.DataFrame()
    

    opts = "-l 0.00001 -k 4 -r 0.02 -t 100 -s 4  --auto-stop"
    acc = []
    files = os.listdir("../train_sample/")
    #files = [files[0]]
    with tqdm(total=len(files), desc='process under the mall ', unit='mall') as pbar:
        for f in files:
            train_file = "../train_sample/" + f
            shop_id_file= "../shop_id/" + f[:-4] + '.csv'
            test_file = "../test_sample/" + f
            train(opts,train_file,val = True)
            
            pred = pd.read_csv(shop_id_file)
            
            train_pred = pred[pred.row_id < 0]
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_pred['shop_id'].values))
            train_pred['label_1'] = lbl.transform(list(train_pred.shop_id.values))  
            num_class=train_pred.label_1.max()+1  
        
            
            predict(train_file)
            train_pred['pred'] = np.loadtxt('./ffm/ffm.preds')
            #调参使用
            x = train_pred.groupby(['row_id']) 
            count = 0
            total = 0
            for i ,group in x:
                total += 1
                dd = group[group.label == 1].shop_id
                group = group.sort_values(by =['pred'],ascending = False)
                temp = group.head(1).shop_id
                if dd.values == temp.values:
                    count += 1
            print count * 1.0 / total
            acc.append(count * 1.0 / total)
            """
            for i, group in x:
    
                index = group.label_1.values
                val = group.pred.values
                
                ypred = np.zeros((220))
                
                ypred[index]  = val
                ypred = ypred.reshape(1,220)
                
                train_end = train_start + 1
                train_res[train_start:train_end,:] = ypred
                train_start = train_end
            """
            predict(test_file)
            test_pred = pred[pred.row_id >= 0]
            test_pred['pred'] = np.loadtxt('./ffm/ffm.preds')
            test_pred['label_1'] = lbl.transform(list(test_pred.shop_id.values))  
            
            x = test_pred.groupby('row_id')
            
            #记录概率和结果
            row = []
            shop = []
            for i, group in x:
                index = group.label_1.values
                val = group.pred.values
                
                ypred = np.zeros((220))
                
                ypred[index]  = val
                ypred = ypred.reshape(1,220)
                
                test_end = test_start + 1
                test_res[test_start:test_end,:] = ypred
                test_start = test_end
                
                row.append(list(set(list(group.row_id)))[0])
                group = group.sort_values(by =['pred'],ascending = False)
                shop.append(group.head(1).shop_id.values[0])
            temp = pd.DataFrame()
            temp['row_id'] = row
            temp['shop_id'] = shop
            result=pd.concat([result,temp])
            cmd = 'rm ' + f + '.bin'
            os.system(cmd) 
            pbar.update(1)
          
               
    result['row_id']=result['row_id'].astype('int')
    result.to_csv(path+'ffm_result.csv',index=False)        
              

