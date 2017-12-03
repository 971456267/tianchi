#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:18:34 2017

@author: flyaway
"""

import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb 
import time as T
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
path='../'
train=pd.read_csv(path+u'训练数据-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'训练数据-ccf_first_round_shop_info.csv')
train=pd.merge(train,shop[['shop_id','mall_id']],how='left',on='shop_id')
test=pd.read_csv(path+u'AB榜测试集-evaluation_public.csv')

#params = {'xgb':0.5,'gbm':0.7} 0.9105
#params = {'xgb':0.4,'gbm':0.6} 0.9106
#params = {'xgb':0.45,'gbm':0.65} 0.9105
params = {'xgb_1':0.4,'xgb_2':0.25,'gbm':0.4} 
xgb_train_1 = np.memmap(path + "result/train_xgb_1.npy", dtype='float32', mode='r', shape=(train.shape[0], 220))
xgb_train_2 = np.memmap(path + "result/train_xgb_2.npy", dtype='float32', mode='r', shape=(train.shape[0], 220))
gbm_train_1 = np.memmap(path + "result/train_gbm_1.npy", dtype='float32', mode='r', shape=(train.shape[0], 220))

xgb_test_1 = np.memmap(path + "result/test_xgb_1.npy", dtype='float32', mode='r', shape=(test.shape[0], 220))
xgb_test_2 = np.memmap(path + "result/test_xgb_2.npy", dtype='float32', mode='r', shape=(test.shape[0], 220))
gbm_test_1 = np.memmap(path + "result/test_gbm_1.npy", dtype='float32', mode='r', shape=(test.shape[0], 220))



def finetuing(xgb_data_1,xgb_data_2,gbm_data,train_or_test = True):#训练的时候打乱了顺序，故有问题
    ypred = params['xgb_1'] * xgb_data_1  + params['xgb_2'] * xgb_data_2 + params['gbm'] * gbm_data
    start = 0
    mall_list=list(set(list(shop.mall_id)))
    
    if train_or_test == True: #test
        total_score = {'True':0,'Total':0,'Score':0.0}
        result=pd.DataFrame()
        for mall in mall_list:
            test1=test[test.mall_id==mall].reset_index(drop=True) 
            train1=train[train.mall_id==mall].reset_index(drop=True) 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train1['shop_id'].values))
            train1['label'] = lbl.transform(list(train1['shop_id'].values)) 
           
            num_class=train1['label'].max()+1
            end  = start + test1.shape[0]
            ypred1 = ypred[start:end,0:num_class]
            ypred1 = np.argmax(ypred1,axis = 1)
            start = end
            test1['label'] = ypred1
            test1['shop_id']=test1['label'].apply(lambda x:lbl.inverse_transform(int(x)))
            r=test1[['row_id','shop_id']]
            result=pd.concat([result,r])
            result['row_id']=result['row_id'].astype('int')
        result['row_id']=result['row_id'].astype('int')
        result.to_csv(path+'result.csv',index=False)
    
    if train_or_test == False: #train
        total_score = {'True':0,'Total':0,'Score':0.0}
        for mall in mall_list:
            train1=train[train.mall_id==mall].reset_index(drop=True) 
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train1['shop_id'].values))
            train1['label'] = lbl.transform(list(train1['shop_id'].values)) 
           
            num_class=train1['label'].max()+1
            end  = start + train1.shape[0]
            ypred1 = ypred[start:end,0:num_class]
            ypred1 = np.argmax(ypred1,axis = 1)
            start = end
            print "  train score:",sum(ypred1 == train1['label'])*100.0 / train1.shape[0]
            total_score['True'] += sum(ypred1 == train1['label'])
            total_score['Total'] += train1.shape[0]
        total_score['Score'] = total_score['True'] * 100.0 / total_score['Total']
        print "  total score",total_score['Score']


finetuing(xgb_test_1,xgb_test_2,gbm_test_1,train_or_test=True)
#finetuing(xgb_train_1,xgb_train_2,gbm_train_1,train_or_test=False)
