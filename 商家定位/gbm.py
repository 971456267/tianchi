#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:48:38 2017

@author: flyaway
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb 
import time as T
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc
path='../'
df=pd.read_csv(path+u'训练数据-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'训练数据-ccf_first_round_shop_info.csv')
test=pd.read_csv(path+u'AB榜测试集-evaluation_public.csv')

train_res = res = np.memmap(path + "result/train_gbm_1.npy", dtype='float32', mode='w+', shape=(df.shape[0], 220))
test_res = np.memmap(path + "result/test_gbm_1.npy", dtype='float32', mode='w+', shape=(test.shape[0], 220))
train_start = 0
test_start = 0

df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
train=pd.concat([df,test])
del df,test
gc.collect()


mall_list=list(set(list(shop.mall_id)))
#mall_list = ['m_6167']
result=pd.DataFrame()
for mall in mall_list:
    train1=train[train.mall_id==mall].reset_index(drop=True)       
    l=[]

    wifi_dict = {}
    for index,row in train1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]]=int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        #添加时间信息
        r['hour'] = int(row['time_stamp'][11:13])
        r['week'] = T.strptime(row['time_stamp'][0:10],'%Y-%m-%d').tm_wday + 1
        l.append(r)    
    
    delate_wifi=[]
    
    #value = wifi_dict.values()
    #value.sort()
    #tmp = value[int(0.05 * len(value))] #5%分位数
    
    for i in wifi_dict:
        
        if wifi_dict[i]< 50:
            delate_wifi.append(i)
    m=[]
    for row in l:
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)
    train1 = pd.concat([train1,pd.DataFrame(m)], axis=1)
    df_train=train1[train1.shop_id.notnull()]
    df_test=train1[train1.shop_id.isnull()]
    del train1
    gc.collect()
    
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1  
   
    feature=[x for x in df_train.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]    
   
    lgbtrain = lgb.Dataset(df_train[feature].values,label = df_train['label'])
    


    param = {'learning_rate':0.01,
             #'num_leaves':31,
             'num_threads':4,
             #'min_data_in_leaf':20,
             #'min_sum_hessian_in_leaf':100,
             'application':'multiclass',
             'metric':'multi_error',
             'num_class':num_class,
             'feature_fraction':0.8,
             'bagging_fraction':0.9,
             'bagging_freq':3}
    
    num_round = 200
    
    bst = lgb.train(param,lgbtrain,num_round,valid_sets=lgbtrain,early_stopping_rounds=10)
    
    ypred = bst.predict(df_train[feature])
    train_end = train_start + ypred.shape[0]
    train_res[train_start:train_end,0:ypred.shape[1]] = ypred
    train_start = train_end
    
    del lgbtrain,df_train
    gc.collect()
    
    ypred = bst.predict(df_test[feature])
    test_end = test_start + ypred.shape[0]
    test_res[test_start:test_end,0:ypred.shape[1]] = ypred
    test_start = test_end
    
    
    ypred = np.argmax(ypred,axis = 1)
    df_test['label'] = ypred
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
 
result['row_id']=result['row_id'].astype('int')
result.to_csv(path+'result/gbm_result_1.csv',index=False)
