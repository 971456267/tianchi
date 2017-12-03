#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:59:27 2017

@author: flyaway
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import seaborn as sns
from get_feature_for_ffm import hav,get_distance_hav,get_feature
import time as T

path = ""
shop_info = pd.read_csv("../训练数据-ccf_first_round_shop_info.csv")
shop_info.columns = ['shop_id', 'category_id', 'shop_longitude', 'shop_latitude','price','mall_id']

train = pd.read_csv("../训练数据-ccf_first_round_user_shop_behavior.csv")
train.columns = ['user_id','shop_id','time_stamp','user_longitude','user_latitude','wifi_infos']

train = pd.merge(train,shop_info,on = 'shop_id',how = 'left')
train['label'] = np.ones(train.shape[0],int)#构造正样本

test = pd.read_csv("../AB榜测试集-evaluation_public.csv")
test.columns = ['row_id','user_id','mall_id','time_stamp','user_longitude','user_latitude','wifi_infos']

mall_list=list(set(list(shop_info.mall_id)))

#mall_list = ['m_3054']
mall_list = [mall_list[27]]
index = 0
for mall in mall_list:
    mall_i = shop_info[shop_info.mall_id==mall].reset_index(drop=True)
    train_pos=train[train.mall_id==mall].reset_index(drop=True) 
    test_i = test[test['mall_id'] == mall].reset_index(drop=True) 
    
    shop_num = len((set(list(mall_i.shop_id))))
    print(shop_num,train_pos.shape,test_i.shape)
    
    train_pos['row_id'] = np.array([-i for i in range(1,train_pos.shape[0] + 1)])
    
    shop_count = train.groupby(['shop_id'])['shop_id'].agg([np.size])
    shop_count = shop_count.reset_index()
    shop_count.columns = ['shop_id','shop_count']
    
    train_pos = pd.merge(train_pos,shop_count,on = 'shop_id',how = 'left')
    
    train_neg = train_pos.drop(['shop_id','category_id', 'shop_longitude', 'shop_latitude','price','label','shop_count'],axis = 1)
    train_neg = pd.merge(train_neg,mall_i,on = 'mall_id',how = 'right')
    train_neg = pd.merge(train_neg,shop_count,on = 'shop_id',how = 'left')
    train_neg['label'] = np.zeros(train_neg.shape[0],int)
    
    cols = train_pos.columns.tolist()
    train_neg = train_neg.loc[:,cols] #是的train_neg 和 train_pos的顺序一样
    
    _cols = [x for x in cols if x not in ['label']]
    train_i = pd.concat([train_pos, train_neg], axis=0,ignore_index = True)
    train_i = train_i.drop_duplicates(_cols[0:-1])#保留出现的第一个重复行，即正样本
    #train_i.to_csv("./train_sample/"+mall+".csv",index=False) #构造的正负样本
    
    test_i = pd.merge(test_i,mall_i,on = 'mall_id',how = 'right')
    test_i = pd.merge(test_i,shop_count,on = 'shop_id',how = 'left')
    #test_i.to_csv("./test_sample/"+mall+".csv",index=False)
    index += 1
    
    print(index,mall,train_i.shape,test_i.shape)
    
    start = T.time()
    print("get feature")
    get_feature(train_i,test_i,mall)
    end = T.time()
    print('cost_time',end-start)

"""    
for file in files:
    file_path = "./mall/" + file
    mall_i = pd.read_csv(file_path)
    mall_name = file[0:-4]
    train_pos = train[train['mall_id'] == mall_name]
    train_pos['row_id'] = np.array([-i for i in range(1,train_pos.shape[0] + 1)])
    
    shop_count = train.groupby(['shop_id'])['shop_id'].agg([np.size])
    shop_count = shop_count.reset_index()
    shop_count.columns = ['shop_id','shop_count']
    
    mall_i =  pd.merge(mall_i,shop_count,on = 'shop_id',how = 'left')#得到每个店铺的销售量
    
    train_pos = pd.merge(train_pos,shop_count,on = 'shop_id',how = 'left')
    
    train_neg = train_pos.drop(['shop_id','category_id', 'shop_longitude', 'shop_latitude','price','label','shop_count'],axis = 1)
    train_neg = pd.merge(train_neg,mall_i,on = 'mall_id',how = 'right')
    train_neg['label'] = np.zeros(train_neg.shape[0],int)
    #表的列的顺序不一样
    cols = list(train_pos)
    train_neg = train_neg.ix[:,cols]
    train_i = pd.concat([train_pos, train_neg], axis=0,ignore_index = True)
    train_i = train_i.drop_duplicates(cols[0:-1])#保留出现的第一个重复行，即正样本
    train_i.to_csv("./train_sample/"+mall_name+".csv
"""