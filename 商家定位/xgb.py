# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:16:39 2017

"""
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb 
import time as T
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Xgboost_Feature import XgboostFeature
import gc
path='../'
df=pd.read_csv(path+u'训练数据-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'训练数据-ccf_first_round_shop_info.csv')
test=pd.read_csv(path+u'AB榜测试集-evaluation_public.csv')

train_res = res = np.memmap(path + "result/train_xgb1.npy", dtype='float32', mode='w+', shape=(df.shape[0], 220))
test_res = np.memmap(path + "result/test_xgb1.npy", dtype='float32', mode='w+', shape=(test.shape[0], 220))
train_start = 0
test_start = 0

df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')


train=pd.concat([df,test])
del df,test
gc.collect()
mall_list=list(set(list(shop.mall_id)))
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
    for i in wifi_dict:
        if wifi_dict[i]<50:
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
    #df_train = df_train
    df_test=train1[train1.shop_id.isnull()]
    del train1
    gc.collect()
    
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1  
    

    feature=[x for x in df_train.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]    
    #clf=XgboostFeature(n_estimators=30)
    ##不切分训练集训练叶子特征模型  返回值 是原特征+新特征
    #X_train, X_test=clf.fit_model(df_train[feature].values, df_train['label'].values,df_test[feature].values)
    """    
    params = {
        'booster':'gbtree',
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 7,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'n_jobs':4,
        'num_class':num_class,
        'silent' : 1,
        'min_child_weight':1,
        'alpha':1,    
        'lambda':1e-5
        }
    xgbtrain = xgb.DMatrix(X_train, df_train['label'])
    del df_train
    gc.collect()
    xgbtest = xgb.DMatrix(X_test)
    watchlist = [ (xgbtrain,'train') ] #, (xgbtrain, 'test')
    num_rounds=100
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=5)
    df_test['label']=model.predict(xgbtest)
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv(path+'sub.csv',index=False)
    model.save_model(path +"model/" + mall + 'xgb')
    """
    #{0.1,100,7,1,0,1,0.9,'multi:softmax',4,0,0,1024}
    model = xgb.XGBClassifier(    
            learning_rate =0.1, #默认0.3    
            n_estimators=100, #树的个数    
            max_depth=7,    
            min_child_weight=1,#默认值  
            gamma=0,    
            subsample=1,    
            colsample_bytree=0.9,    
            objective= 'multi:softmax', #逻辑回归损失函数    
            nthread=4,  #cpu线程数    
            #scale_pos_weight=1,    
            reg_alpha=1e-6,    
            reg_lambda=1,   
            seed=1024,
            #missing=-999
            )    
    #X_train, X_valid, y_train, y_valid = train_test_split(df_train[feature], df_train['label'], test_size=0.1, random_state=0)          
    #model.fit(X_train, y_train,eval_set = [(X_train,y_train),(X_valid, y_valid)], eval_metric = 'merror',early_stopping_rounds = 15)  
    model.fit(df_train[feature], df_train['label'], eval_set=[(df_train[feature], df_train['label'])],eval_metric='merror',early_stopping_rounds=10)

    ypred = model.predict_proba(df_train[feature])
    train_end = train_start + ypred.shape[0]
    train_res[train_start:train_end,0:ypred.shape[1]] = ypred
    train_start = train_end
    
    del df_train
    gc.collect()
    
    ypred = model.predict_proba(df_test[feature])
    test_end = test_start + ypred.shape[0]
    test_res[test_start:test_end,0:ypred.shape[1]] = ypred
    test_start = test_end
    
    
    df_test['label']=model.predict(df_test[feature])
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])

result['row_id']=result['row_id'].astype('int')
result.to_csv(path+'result/xgb_result_1.csv',index=False)
