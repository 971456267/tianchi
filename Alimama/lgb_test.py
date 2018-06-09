#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import pandas as pd
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
import math
def encode_feature(values):
    uniq = values.unique()
    mapping = dict(zip(uniq,range(1,len(uniq) + 1)))
    return values.map(mapping)


def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def get_week(dt):
    rt = (time.strptime(dt[0:10],'%Y-%m-%d').tm_wday + 1)
  
    is_weekday = 1 if rt >=6 else 0
    return is_weekday

def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['is_week'] = data.time.apply(get_week)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data

from  collections import Counter
def process_property_feature(data,sub_data):
    #for all 
    property_ = []
    for item in data['item_property_list']:
        property_.extend(item.split(";"))
    
    most_count = Counter(property_).most_common(100) 
    property_ = []
    for item in most_count:
        property_.append(item[0])  
    property_ = set(property_)
    
    most_property_num = []
    most_proerty = []
    for item in sub_data['item_property_list']:
        tmp = set(item.split(";"))
        tmp = tmp & property_
        most_property_num.append(len(tmp))  
        most_proerty.append(";".join(list(tmp)))
    sub_data['most_property_num'] = most_property_num
    sub_data['most_property'] = most_proerty
    sub_data['most_property'] = encode_feature(sub_data['most_property'])
    
    #for trade
    trade_data = data.loc[data.is_trade == 1]
    property_ = []
    for item in trade_data['item_property_list']:
        property_.extend(item.split(";"))
        
    most_count = Counter(property_).most_common(100) 
    property_ = []  
    for item in most_count:
        property_.append(item[0])
    property_ = set(property_)
    
    most_count = Counter(property_).most_common()
    most_property_num = []
    most_proerty = []
    for item in sub_data['item_property_list']:
        tmp = set(item.split(";"))
        tmp = tmp & property_
        most_property_num.append(len(tmp))
        most_proerty.append(";".join(list(tmp)))
    sub_data['trade_most_property_num'] = most_property_num
    sub_data['trade_most_property'] = most_proerty
    sub_data['trade_most_property'] = encode_feature(sub_data['trade_most_property']) 
    
    sub_data['item_property_num'] = sub_data['item_property_list'].apply(lambda x:len(x.split(";")))
    del sub_data['item_property_list']
    return sub_data
    
def process_list_feature(data):
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else "-1"
        )
        data['category_%d'%(i)] = data['category_%d'%(i)].astype(int)
    del data['item_category_list']
    
    for i in range(3):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(
            lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else "-1"
        )
        data['predict_category_%d'%(i)] = data['predict_category_%d'%(i)].astype(int)
    del data['predict_category_property']
    
    return data

def count_feat(all_data,data):

    #基于all
    item_count = all_data.groupby(['item_id'],as_index = False).size().reset_index()  
    item_count.columns = ['item_id','item_count']
    #item_count['item_count'] = item_count['item_count'].apply(np.log)
    
    user_count = all_data.groupby(['user_id'],as_index = False).size().reset_index()
    user_count.columns = ['user_id','user_count']
    #user_count['user_count'] = user_count['user_count'].apply(np.log)
    
    shop_count = all_data.groupby(['shop_id'],as_index = False).size().reset_index()
    shop_count.columns = ['shop_id','shop_count']
    #shop_count['shop_count'] = shop_count['shop_count'].apply(np.log)
    
    cat_0_count = all_data.groupby(['category_0'],as_index = False).size().reset_index()
    cat_0_count.columns = ['category_0','cat_0_count']
    
    cat_1_count = all_data.groupby(['category_1'],as_index = False).size().reset_index()
    cat_1_count.columns = ['category_1','cat_1_count']
    
    cat_2_count = all_data.groupby(['category_2'],as_index = False).size().reset_index()
    cat_2_count.columns = ['category_2','cat_2_count']
    
    ####基于trade
    trade_data = all_data.loc[all_data.is_trade == 1]
    
    trade_item_count = trade_data.groupby(['item_id'],as_index = False).size().reset_index()  
    trade_item_count.columns = ['item_id','trade_item_count'] 
    
    trade_user_count = trade_data.groupby(['user_id'],as_index = False).size().reset_index()  
    trade_user_count.columns = ['user_id','trade_user_count']
    
    trade_shop_count = trade_data.groupby(['shop_id'],as_index = False).size().reset_index()  
    trade_shop_count.columns = ['shop_id','trade_shop_count']
    
    trade_cat_0_count = trade_data.groupby(['category_0'],as_index = False).size().reset_index()
    trade_cat_0_count.columns = ['category_0','trade_cat_0_count']
    
    trade_cat_1_count = trade_data.groupby(['category_1'],as_index = False).size().reset_index() 
    trade_cat_1_count.columns = ['category_1','trade_cat_1_count']
    
    trade_cat_2_count = trade_data.groupby(['category_2'],as_index = False).size().reset_index()  
    trade_cat_2_count.columns = ['category_2','trade_cat_2_count']
    
    trade_item = pd.merge(item_count,trade_item_count,on = ['item_id'],how = 'left')
    trade_user = pd.merge(user_count,trade_user_count,on = ['user_id'],how = 'left')
    trade_shop = pd.merge(shop_count,trade_shop_count,on = ['shop_id'],how = 'left')
    
    trade_cat_0 = pd.merge(cat_0_count,trade_cat_0_count,on = ['category_0'],how = 'left')
    trade_cat_1 = pd.merge(cat_1_count,trade_cat_1_count,on = ['category_1'],how = 'left')
    trade_cat_2 = pd.merge(cat_2_count,trade_cat_2_count,on = ['category_2'],how = 'left')
    
    def do_something(x, y):
        try:
            return y/(float(x + 10))
        except:
            return 0
    trade_item['item_ctr'] = map(lambda x, y: do_something(x, y) , trade_item['item_count'], trade_item['trade_item_count'])
    trade_user['user_ctr'] = map(lambda x, y: do_something(x, y) , trade_user['user_count'], trade_user['trade_user_count'])
    trade_shop['shop_ctr'] = map(lambda x, y: do_something(x, y) , trade_shop['shop_count'], trade_shop['trade_shop_count'])
    trade_cat_0['cat_0_ctr'] = map(lambda x, y: do_something(x, y) , trade_cat_0['category_0'], trade_cat_0['trade_cat_0_count'])
    trade_cat_1['cat_1_ctr'] = map(lambda x, y: do_something(x, y) , trade_cat_1['category_1'], trade_cat_1['trade_cat_1_count'])
    trade_cat_2['cat_2_ctr'] = map(lambda x, y: do_something(x, y) , trade_cat_2['category_2'], trade_cat_2['trade_cat_2_count'])
    data = pd.merge(data,trade_item,on=['item_id'],how='left')
    data = pd.merge(data,trade_user,on=['user_id'],how='left')
    data = pd.merge(data,trade_shop,on=['shop_id'],how='left')
    data = pd.merge(data,trade_cat_0,on = ['category_0'],how = 'left')
    data = pd.merge(data,trade_cat_1,on = ['category_1'],how = 'left')
    data = pd.merge(data,trade_cat_2,on = ['category_2'],how = 'left')
    
    data = data.fillna(-1)
    

    return data

def get_whether_trade(data,sub_data):
    #获取item_user对中是否已经购买过，有了理由相信如果已经购买过了，之后不会交易
    #或者是之前一直没有交易，但是看了很多次，代表有更大的可能购买
    item_user = data.groupby(['item_id','user_id'],as_index = False)
    show = item_user['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['item_id','user_id','show']
    sub_data = pd.merge(sub_data,show,on=['item_id','user_id'],how = 'left')
    
    trade_data = data.loc[data.is_trade == 1]
    trade_data = trade_data[['item_id','user_id','context_timestamp']]
    trade_data.sort_values(by="context_timestamp" , ascending=False)#降序
    trade_data.columns = ['item_id','user_id','trade_time']
    trade_data.drop_duplicates(subset=['item_id','user_id'], keep='first', inplace=True) 
    
    sub_data = pd.merge(sub_data,trade_data,on=['item_id','user_id'],how = 'left')
  
    #sub_data['trade_already'] = map(lambda x,y:  0 if math.isnan(y) else (1 if x > y else (2 if x = y else 0 )) , sub_data['context_timestamp'],sub_data['trade_time'])
    sub_data['trade_no'] = map(lambda x,y:  1 if math.isnan(y) else (1 if x <= y else 0) , sub_data['context_timestamp'],sub_data['trade_time'])
    del sub_data['trade_time']
    del sub_data['context_timestamp']
    return sub_data

def encoder_id_feature(train,test,cat_feat,online = False):
    data = pd.concat([train,test])
    for feat in cat_feat:
        data[feat] = encode_feature(data[feat])
    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    else:
        train = data.loc[data.day < 25]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 25]  # 暂时先使用第24天作为验证集
    return train,test

def cal_sum(train,test,online = False):
    data = pd.concat([train,test])
    data['ss'] = map(lambda x:1,data.is_trade)
    data['sum_Times']=data.groupby(['item_id','user_id'])['ss'].cumsum()
    del data['ss']
    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    else:
        train = data.loc[data.day < 25]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 25]  # 暂时先使用第24天作为验证集
    return train,test

if __name__ == "__main__":
    #id特征
    id_feat = ['item_id','user_id','item_brand_id', 'item_city_id',
             'user_occupation_id','context_page_id', 'shop_id',
             'category_0','category_1','category_2',
             'predict_category_0','predict_category_1','predict_category_2',
             'most_property','trade_most_property','hour']
    online = False# 这里用来标记是 线下验证 还是 在线提交

    data = pd.read_csv('./round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = convert_data(data)
    data = process_list_feature(data)
    
    if online == False:
        
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
        all_data = train.copy()
        
        train = process_property_feature(all_data,train)
        train = count_feat(all_data,train)
        train = get_whether_trade(all_data,train)
        test = process_property_feature(all_data,test)
        test = count_feat(all_data,test)
        test =get_whether_trade(all_data,test)
        
        train,test = encoder_id_feature(train,test,id_feat,online)
        train,test = cal_sum(train,test,online)
    elif online == True:
        train = data.copy()
    
        test = pd.read_csv('./round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)
        test = process_list_feature(test)
        
        test = process_property_feature(data,train)
        test = count_feat(data,test)
        test =get_whether_trade(data,test)
        
        train = process_property_feature(data,train)
        train = count_feat(data,train)
        train = get_whether_trade(data,train)
        
        train,test = encoder_id_feature(train,test,id_feat,online)
        train,test = cal_sum(train,test,online)
        
    features = ['item_id','user_id','item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level','user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                'category_0','category_1','category_2',
                'predict_category_0','predict_category_1','predict_category_2',
                'most_property_num','most_property','trade_most_property_num','trade_most_property','item_property_num',
                #'item_count','user_count','shop_count',
                #'item_ctr','user_ctr','shop_ctr',
                'cat_0_count','cat_1_count','cat_2_count',
                'cat_0_ctr','cat_1_ctr','cat_2_ctr',
                'show','trade_no','shop_count','is_week','sum_Times'] #trade_no,trade_already
    #容易过拟合的特征
    feat1 = ['item_count','user_count',
             'item_ctr','user_ctr','shop_ctr',
             ]
    
    cat_feat = id_feat
    cat_feat.extend(['user_gender_id','hour','is_week']) #trade_already
    target = ['is_trade']
    
    clf = lgb.LGBMClassifier(num_leaves=31, 
                                 learning_rate = 0.05,
                                 max_depth=9, 
                                 n_estimators=200, 
                                 n_jobs=20,
                                 reg_lambda =2,
                                 objective="binary:logistic",
                                 subsample=1, 
                                 colsample_bytree=0.5,
                                 min_child_weight = 15,
                                 min_child_samples = 10,
                                 random_state= 2048)
    
    if online == False:
        clf.fit(train[features], train[target], eval_set=[(train[features], train[target]),(test[features], test[target])],
        eval_metric = "logloss",feature_name=features,categorical_feature = cat_feat,early_stopping_rounds = 10)
  
        train['lgb_predict'] = clf.predict_proba(train[features])[:, 1]
        print('train log_loss',log_loss(train[target], train['lgb_predict']))
        test['lgb_predict'] = clf.predict_proba(test[features])[:, 1]
        print('test log_loss',log_loss(test[target], test['lgb_predict']))
    else:
        clf.fit(train[features], train[target],feature_name=features,categorical_feature = cat_feat)
        
        train['predicted_score'] = clf.predict_proba(train[features])[:, 1]
        print('train log_loss',log_loss(train[target], train['predicted_score']))
        
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        
        train[['instance_id','is_trade', 'predicted_score']].to_csv('lgb_train_baseline.csv', index=False,sep=' ')
        test[['instance_id', 'predicted_score']].to_csv('lgb_test_baseline.csv', index=False,sep=' ')
# =============================================================================
# ('train log_loss', 0.08171284013288146)
# ('test log_loss', 0.0817894367562419) #没有list 属性
# =============================================================================
# =============================================================================
# ('train log_loss', 0.08098094195241205)
# ('test log_loss', 0.08172050907917039)　有list属性
# =============================================================================
# =============================================================================
# ('train log_loss', 0.08180761167149365)
# ('test log_loss', 0.08169701615030701)有list属性
# =============================================================================
