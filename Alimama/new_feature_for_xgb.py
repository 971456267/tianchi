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
    data['half_hour'] = data.time.apply(lambda x : 1 if int(x[14:16]) >= 30 else 0)
    data['half_hour'] = map(lambda x,y:  str(x) + str(y), data['hour'],data['half_hour'])
    data['half_hour'] = encode_feature(data.half_hour)
    """
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',on=['user_id', 'day', 'hour'])   
    """   
    user_query_day_half_hour = data.groupby(['user_id', 'day', 'half_hour']).size().reset_index().rename(columns={0: 'user_query_day_half_hour'})
    data = pd.merge(data, user_query_day_half_hour, 'left',on=['user_id', 'day', 'half_hour'])   
    return data

from  collections import Counter
def convert_cat(data):
    #所有商品的一级类目都一样，二级类目为13个，三级类目2,还有一个为-1
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else "-1")
        data['category_%d'%(i)] = data['category_%d'%(i)].astype(int)
    #del data['item_category_list']
    return data

def convert_pro(data):
    #for all 
    property_ = []
    trade_data = data.loc[data.is_trade == 1]
    for item in trade_data['item_property_list']:
        property_.extend(item.split(";"))
    
    most_count = Counter(property_).most_common(10) #1000,出现在前1000的属性 
    property_ = []
    for item in most_count:
        property_.append(item[0])  
    
    property_set = set(property_)
    
    
    property_dict = collections.defaultdict(lambda : -1)
    for i,item in enumerate(property_):
        property_dict[item] = i #给属性编号  
    #most_proerty = []
    np_maxtrix = np.zeros((data.shape[0],10))
    for i, item in enumerate(data['item_property_list']):
        tmp = set(item.split(";"))
        tmp = tmp & property_set
        sub = []
        tmp = list(tmp)
        
        sub = [property_dict[x] for x in tmp]
        np_maxtrix[i,:] = [ 1 if x in sub else 0 for x in np_maxtrix[i,:]]
        #most_proerty.append(sub)
        #most_proerty.append(";".join(list(tmp)))
    #data['most_proerty' + str(i)] = most_proerty
    for i in range(10):
        data['most_proerty_' + str(i)] = np_maxtrix[:,i]
    return data

import collections,gc
def convert_predict(data):
    predict_cat = []
    predict_pro = []

    for item in data['predict_category_property']:
        xx = item.split(";")
        for tmp in xx:
            try:          
                if tmp.split(":") [1] == "-1":
                    predict_cat.append(tmp.split(":")[0])
                else:
                    predict_cat.append(tmp.split(":")[0])
                    pro_list = tmp.split(":") [1].split(",")
                    predict_pro.extend(pro_list)
            except:
                continue
    most_count_predict_cat = Counter(predict_cat).most_common(30)
    most_count_predict_pro = Counter(predict_pro).most_common(25)

    gc.collect()
    predict_cat = []
    for item in most_count_predict_cat:
        predict_cat.append(item[0])  
    predict_cat_set = set(predict_cat)
    
    predict_pro = []
    for item in most_count_predict_pro:
        predict_pro.append(item[0])  
    predict_pro_set = set(predict_pro)
    
    predict_cat_dict = collections.defaultdict(lambda : -1)
    predict_pro_dict = collections.defaultdict(lambda : -1)
    
    for i,item in enumerate(predict_cat):
        predict_cat_dict[item] = i #给属性编号  
        
    for i,item in enumerate(predict_pro):
        predict_pro_dict[item] = i #给属性编号  
        

    cat_maxtrix = np.zeros((data.shape[0],30))
    pro_maxtrix = np.zeros((data.shape[0],25))
    for i, item in enumerate(data['predict_category_property']):
        sub_cat = []
        sub_pro = []
        xx = item.split(";")
        for tmp in xx:
            try:          
                if tmp.split(":") [1] == "-1":
                    sub_cat.append(tmp.split(":")[0])
                else:
                    
                    sub_cat.append(tmp.split(":")[0])
                    pro_list = tmp.split(":") [1].split(",")
                    sub_pro.extend(pro_list)
            except:
                continue
            
        
        #print sub_pro
        sub_cat = list(set(sub_cat) & (predict_cat_set))
        sub_pro = list(set(sub_pro) & (predict_pro_set))
        
        sub_cat = [predict_cat_dict[x] for x in sub_cat]
        sub_pro = [predict_pro_dict[x] for x in sub_pro]


        cat_maxtrix[i,:] = [ 1 if x in sub_cat else 0 for x in cat_maxtrix[i,:]]
        pro_maxtrix[i,:] = [ 1 if x in sub_pro else 0 for x in pro_maxtrix[i,:]]

    for i in range(30):
        data['most_predict_cat_' + str(i)] = cat_maxtrix[:,i]
    for i in range(25):
        data['most_predict_pro_' + str(i)] = pro_maxtrix[:,i]   
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
    
    cat_1_count = all_data.groupby(['category_1'],as_index = False).size().reset_index()
    cat_1_count.columns = ['category_1','cat_1_count']
    
    ####基于trade
    trade_data = all_data.loc[all_data.is_trade == 1]
    
    trade_item_count = trade_data.groupby(['item_id'],as_index = False).size().reset_index()  
    trade_item_count.columns = ['item_id','trade_item_count'] 
    
    trade_user_count = trade_data.groupby(['user_id'],as_index = False).size().reset_index()  
    trade_user_count.columns = ['user_id','trade_user_count']
    
    trade_shop_count = trade_data.groupby(['shop_id'],as_index = False).size().reset_index()  
    trade_shop_count.columns = ['shop_id','trade_shop_count']
        
    trade_cat_1_count = trade_data.groupby(['category_1'],as_index = False).size().reset_index() 
    trade_cat_1_count.columns = ['category_1','trade_cat_1_count']
    
  
    
    trade_item = pd.merge(item_count,trade_item_count,on = ['item_id'],how = 'left')
    trade_user = pd.merge(user_count,trade_user_count,on = ['user_id'],how = 'left')
    trade_shop = pd.merge(shop_count,trade_shop_count,on = ['shop_id'],how = 'left')   
    trade_cat_1 = pd.merge(cat_1_count,trade_cat_1_count,on = ['category_1'],how = 'left')

    
    def do_something(x, y):
        #a = 10
        #b = 100
        try:
            return (y)/(float(x+10))
        except:
            return 0
    trade_item['item_ctr'] = map(lambda x, y: do_something(x, y) , trade_item['item_count'], trade_item['trade_item_count'])
    trade_user['user_ctr'] = map(lambda x, y: do_something(x, y) , trade_user['user_count'], trade_user['trade_user_count'])
    trade_shop['shop_ctr'] = map(lambda x, y: do_something(x, y) , trade_shop['shop_count'], trade_shop['trade_shop_count'])
    trade_cat_1['cat_1_ctr'] = map(lambda x, y: do_something(x, y) , trade_cat_1['category_1'], trade_cat_1['trade_cat_1_count'])

    data = pd.merge(data,trade_item,on=['item_id'],how='left')
    data = pd.merge(data,trade_user,on=['user_id'],how='left')
    data = pd.merge(data,trade_shop,on=['shop_id'],how='left')
    data = pd.merge(data,trade_cat_1,on = ['category_1'],how = 'left')
   

    
    #data = data.fillna(-1)fillna()是个迷，慎用
    return data

def count(data):
    item_count = data.groupby(['item_id'],as_index = False).size().reset_index()  
    item_count.columns = ['item_id','item_count']
    #item_count['item_count'] = item_count['item_count'].apply(np.log)
    
    user_count = data.groupby(['user_id'],as_index = False).size().reset_index()
    user_count.columns = ['user_id','user_count']
    #user_count['user_count'] = user_count['user_count'].apply(np.log)
    
    shop_count = data.groupby(['shop_id'],as_index = False).size().reset_index()
    shop_count.columns = ['shop_id','shop_count']
    #shop_count['shop_count'] = shop_count['shop_count'].apply(np.log)    
    
    cat_1_count = data.groupby(['category_1'],as_index = False).size().reset_index()
    cat_1_count.columns = ['category_1','cat_1_count']
    
    item_brand_id_count = data.groupby(['item_brand_id'],as_index = False).size().reset_index()
    item_brand_id_count.columns = ['item_brand_id','item_brand_id_count']
    
    item_city_id_count = data.groupby(['item_city_id'],as_index = False).size().reset_index()
    item_city_id_count.columns = ['item_city_id','item_city_id_count']

    
    item_pv_level_count = data.groupby(['item_pv_level'],as_index = False).size().reset_index()
    item_pv_level_count.columns = ['item_pv_level','item_pv_level_count']
    
    user_age_level_count = data.groupby(['user_age_level'],as_index = False).size().reset_index()
    user_age_level_count.columns = ['user_age_level','user_age_level_count']
    
    item_sales_level_count = data.groupby(['item_sales_level'],as_index = False).size().reset_index()
    item_sales_level_count.columns = ['item_sales_level','item_sales_level_count']
    

    item_price_level_count = data.groupby(['item_price_level'],as_index = False).size().reset_index()
    item_price_level_count.columns = ['item_price_level','item_price_level_count']
    
    item_sales_level_count = data.groupby(['item_sales_level'],as_index = False).size().reset_index()
    item_sales_level_count.columns = ['item_sales_level','item_sales_level_count']
    
    item_collected_level_count = data.groupby(['item_collected_level'],as_index = False).size().reset_index()
    item_collected_level_count.columns = ['item_collected_level','item_collected_level_count']
    
    user_gender_id_count = data.groupby(['user_gender_id'],as_index = False).size().reset_index()
    user_gender_id_count.columns = ['user_gender_id','user_gender_id_count']
    
    user_occupation_id_count = data.groupby(['user_occupation_id'],as_index = False).size().reset_index()
    user_occupation_id_count.columns = ['user_occupation_id','user_occupation_id_count']
      
    user_star_level_count = data.groupby(['user_star_level'],as_index = False).size().reset_index()
    user_star_level_count.columns = ['user_star_level','user_star_level_count']
    
    context_page_id_count = data.groupby(['context_page_id'],as_index = False).size().reset_index()
    context_page_id_count.columns = ['context_page_id','context_page_id_count']
    
    shop_review_num_level_count = data.groupby(['shop_review_num_level'],as_index = False).size().reset_index()
    shop_review_num_level_count.columns = ['shop_review_num_level','shop_review_num_level_count']
    
        
    shop_star_level_count = data.groupby(['shop_star_level'],as_index = False).size().reset_index()
    shop_star_level_count.columns = ['shop_star_level','shop_star_level_count']
    

    data = pd.merge(data,item_count,on=['item_id'],how='left')
    data = pd.merge(data,user_count,on=['user_id'],how='left')
    data = pd.merge(data,shop_count,on=['shop_id'],how='left')
    data = pd.merge(data,item_brand_id_count,on = ['item_brand_id'],how = 'left')
    data = pd.merge(data,item_city_id_count,on = ['item_city_id'],how = 'left')
    data = pd.merge(data,cat_1_count,on = ['category_1'],how = 'left')
    data = pd.merge(data,item_pv_level_count,on=['item_pv_level'],how = 'left')
    data = pd.merge(data,user_age_level_count,on = ['user_age_level'],how = 'left')
    data = pd.merge(data,item_price_level_count,on = ['item_price_level'],how = 'left')
    data = pd.merge(data,item_sales_level_count,on = ['item_sales_level'],how = 'left')
    data = pd.merge(data,item_collected_level_count,on = ['item_collected_level'],how = 'left')
    data = pd.merge(data,user_gender_id_count,on = ['user_gender_id'],how = 'left') 
    data = pd.merge(data,user_occupation_id_count,on = ['user_occupation_id'],how = 'left')
    data = pd.merge(data,user_star_level_count,on = ['user_star_level'],how = 'left')
    data = pd.merge(data,context_page_id_count,on = ['context_page_id'],how = 'left')
    data = pd.merge(data,shop_review_num_level_count,on = ['shop_review_num_level'],how = 'left')
    data = pd.merge(data,shop_star_level_count,on = ['shop_star_level'],how = 'left')
   
    return data

    
  
#cross_feature_two_one_day,统计两个特征的在一天内的融合
def cross_feature_two(data):
    #统计用户点击个多少种item
    user_item = data.groupby(['user_id','item_id'],as_index = False)
    show = user_item['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_id','user_item']
    data = pd.merge(data,show,on=['user_id','item_id'],how = 'left')
    
    #统计用户点击个多少种category_1
    user_cat_1 = data.groupby(['user_id','category_1'],as_index = False)
    show = user_cat_1['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','category_1','user_cat_1']
    data = pd.merge(data,show,on=['user_id','category_1'],how = 'left')
    
    #统计用户点击个多少种shop
    user_shop = data.groupby(['user_id','shop_id'],as_index = False)
    show = user_shop['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','shop_id','user_shop']
    data = pd.merge(data,show,on=['user_id','shop_id'],how = 'left')
    
    #统计用户点击个多少种item_brand_id
    user_brand = data.groupby(['user_id','item_brand_id'],as_index = False)
    show = user_brand['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_brand_id','user_brand']
    data = pd.merge(data,show,on=['user_id','item_brand_id'],how = 'left')
    
    #统计用户点击个多少种item_city_id
    user_city= data.groupby(['user_id','item_city_id'],as_index = False)
    show = user_city['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_city_id','user_city']
    data = pd.merge(data,show,on=['user_id','item_city_id'],how = 'left')
    
    #统计用户点击个多少种item_price_leve
    user_price= data.groupby(['user_id','item_price_level'],as_index = False)
    show = user_price['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_price_level','user_price']
    data = pd.merge(data,show,on=['user_id','item_price_level'],how = 'left')
    
    #统计用户点击个多少种item_sales_level
    user_sales= data.groupby(['user_id','item_sales_level'],as_index = False)
    show = user_sales['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_sales_level','user_sales']
    data = pd.merge(data,show,on=['user_id','item_sales_level'],how = 'left')
    
    #统计用户点击个多少种item_collected_level
    user_collected= data.groupby(['user_id','item_collected_level'],as_index = False)
    show = user_collected['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_collected_level','user_collected']
    data = pd.merge(data,show,on=['user_id','item_collected_level'],how = 'left')
    
    #统计用户点击个多少种item_collected_level
    user_pv= data.groupby(['user_id','item_pv_level'],as_index = False)
    show = user_pv['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','item_pv_level','user_pv']
    data = pd.merge(data,show,on=['user_id','item_pv_level'],how = 'left')
    
    
    user_gender= data.groupby(['user_id','user_gender_id'],as_index = False)
    show = user_gender['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','user_gender_id','user_gender']
    data = pd.merge(data,show,on=['user_id','user_gender_id'],how = 'left')
    
    user_age= data.groupby(['user_id','user_age_level'],as_index = False)
    show = user_age['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','user_age_level','user_age']
    data = pd.merge(data,show,on=['user_id','user_age_level'],how = 'left')
    
    user_occupation= data.groupby(['user_id','user_occupation_id'],as_index = False)
    show = user_occupation['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','user_occupation_id','user_occupation']
    data = pd.merge(data,show,on=['user_id','user_occupation_id'],how = 'left')
    
    user_star= data.groupby(['user_id','user_star_level'],as_index = False)
    show = user_star['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','user_star_level','user_star']
    data = pd.merge(data,show,on=['user_id','user_star_level'],how = 'left')
    
    
    user_context_page_id = data.groupby(['user_id','context_page_id'],as_index = False)
    show = user_context_page_id['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','context_page_id','user_context_page_id']
    data = pd.merge(data,show,on=['user_id','context_page_id'],how = 'left')
    
    user_day= data.groupby(['user_id','day'],as_index = False)
    show = user_day['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','user_day']
    data = pd.merge(data,show,on=['user_id','day'],how = 'left') 
    
    user_hour= data.groupby(['user_id','hour'],as_index = False)
    show = user_hour['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','hour','user_hour']
    data = pd.merge(data,show,on=['user_id','hour'],how = 'left')
    
    user_half_hour= data.groupby(['user_id','half_hour'],as_index = False)
    show = user_half_hour['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','half_hour','user_half_hour']
    data = pd.merge(data,show,on=['user_id','half_hour'],how = 'left')
    
    user_week= data.groupby(['user_id','is_week'],as_index = False)
    show = user_week['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','is_week','user_week']
    data = pd.merge(data,show,on=['user_id','is_week'],how = 'left')
    
    
    user_review= data.groupby(['user_id','shop_review_num_level'],as_index = False)
    show = user_review['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','shop_review_num_level','user_review']
    data = pd.merge(data,show,on=['user_id','shop_review_num_level'],how = 'left')
    
    """
    #统计用户点击个多少种shop_review_positive_rate
    user_review_pos= data.groupby(['user_id','shop_review_positive_rate'],as_index = False)
    show = user_review_pos['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','shop_review_positive_rate','user_review_pos']
    data = pd.merge(data,show,on=['user_id','shop_review_positive_rate'],how = 'left')
    
   
    #统计用户点击个多少种shop_review_positive_rate
    user_star= data.groupby(['user_id','shop_star_level'],as_index = False)
    show = user_star['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','shop_star_level','user_star']
    data = pd.merge(data,show,on=['user_id','shop_star_level'],how = 'left')
    """
    return data

def cross_feature_two_one_day(data):
    #统计用户在一天内点击个多少种item
    user_day_item = data.groupby(['user_id','day','item_id'],as_index = False)
    show = user_day_item['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','item_id','user_day_item']
    data = pd.merge(data,show,on=['user_id','day','item_id'],how = 'left')
    
    #统计用户在一天内点击个多少种category_1
    user_day_cat_1 = data.groupby(['user_id','day','category_1'],as_index = False)
    show = user_day_cat_1['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','category_1','user_day_cat_1']
    data = pd.merge(data,show,on=['user_id','category_1','day'],how = 'left')
    
    #统计用户在一天内点击个多少种shop
    user_day_shop = data.groupby(['user_id','day','shop_id'],as_index = False)
    show = user_day_shop['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','shop_id','user_day_shop']
    data = pd.merge(data,show,on=['user_id','shop_id','day'],how = 'left')
    
    #统计用户在一天内点击个多少种item_brand_id
    user_day_brand = data.groupby(['user_id','day','item_brand_id'],as_index = False)
    show = user_day_brand['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','item_brand_id','user_day_brand']
    data = pd.merge(data,show,on=['user_id','day','item_brand_id'],how = 'left')
    
    return data


def cross_feature_two_one_hour(data):
    #统计用户在一小时内点击个多少种item
    user_hour_item = data.groupby(['user_id','day','hour','item_id'],as_index = False)
    show = user_hour_item['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','hour','item_id','user_hour_item']
    data = pd.merge(data,show,on=['user_id','day','hour','item_id'],how = 'left')
    
    #统计用户在一小时内点击个多少种category_1
    user_hour_cat_1 = data.groupby(['user_id','day','hour','category_1'],as_index = False)
    show = user_hour_cat_1['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','hour','category_1','user_hour_cat_1']
    data = pd.merge(data,show,on=['user_id','category_1','hour','day'],how = 'left')
    
    #统计用户在小时内点击个多少种shop
    user_hour_shop = data.groupby(['user_id','day','hour','shop_id'],as_index = False)
    show = user_hour_shop['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','hour','shop_id','user_hour_shop']
    data = pd.merge(data,show,on=['user_id','shop_id','day','hour'],how = 'left')
    
    #统计用户在一小时内点击个多少种item_brand_id
    user_hour_brand = data.groupby(['user_id','day','hour','item_brand_id'],as_index = False)
    show = user_hour_brand['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','hour','item_brand_id','user_hour_brand']
    data = pd.merge(data,show,on=['user_id','day','hour','item_brand_id'],how = 'left')
    return data

def cross_feature_two_half_hour(data):
    #统计用户在一小时内点击个多少种item
    user_half_hour_item = data.groupby(['user_id','day','half_hour','item_id'],as_index = False)
    show = user_half_hour_item['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','half_hour','item_id','user_half_hour_item']
    data = pd.merge(data,show,on=['user_id','day','half_hour','item_id'],how = 'left')
    
    #统计用户在一小时内点击个多少种category_1
    user_half_hour_cat_1 = data.groupby(['user_id','day','half_hour','category_1'],as_index = False)
    show = user_half_hour_cat_1['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','half_hour','category_1','user_half_hour_cat_1']
    data = pd.merge(data,show,on=['user_id','category_1','half_hour','day'],how = 'left')
    
    #统计用户在小时内点击个多少种shop
    user_half_hour_shop = data.groupby(['user_id','day','half_hour','shop_id'],as_index = False)
    show = user_half_hour_shop['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','half_hour','shop_id','user_half_hour_shop']
    data = pd.merge(data,show,on=['user_id','shop_id','day','half_hour'],how = 'left')
    
    #统计用户在一小时内点击个多少种item_brand_id
    user_half_hour_brand = data.groupby(['user_id','day','half_hour','item_brand_id'],as_index = False)
    show = user_half_hour_brand['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['user_id','day','half_hour','item_brand_id','user_half_hour_brand']
    data = pd.merge(data,show,on=['user_id','day','half_hour','item_brand_id'],how = 'left')
    return data

def first_and_next(data,date):
    #data.sort_values(by="context_timestamp" , ascending=True)#升序
    data['new_trade'] = map(lambda x,y:  str(x) if y  < date  else ( str(0) ) , data.is_trade,data.day)
    def help(arr):
        arr = arr.sort_values(by=["context_timestamp"],ascending = True)
        arr['diff_1'] = arr.context_timestamp.diff().div(60).abs()
        arr['diff_2'] = arr.context_timestamp.diff(-1).div(60).abs()
        arr['diff_3'] = (arr.context_timestamp - arr.context_timestamp.max()).div(60).abs()
        arr['diff_4'] = (arr.context_timestamp - arr.context_timestamp.min()).div(60).abs()
        arr['history'] = arr.new_trade.cumsum()
        arr['history'] = map(lambda x: x[:-1],arr.history)
        return arr[['instance_id','diff_1','diff_2','diff_3','diff_4','history']]
    
    arr = data.groupby(['user_id','day']).apply(help)
    arr = arr.reset_index(drop=True)
    data = pd.merge(data,arr,on=['instance_id'],how = 'left')
    del data['new_trade']
    return data

def first_and_next_2(data,date):
    #data.sort_values(by="context_timestamp" , ascending=True)#升序
    data['new_trade'] = map(lambda x,y:  str(x) if y  < date  else ( str(0) ) , data.is_trade,data.day)
    def help(arr):
        arr = arr.sort_values(by=["context_timestamp"],ascending = True)
        arr['all_diff_1'] = arr.context_timestamp.diff().div(60).abs()
        arr['all_diff_2'] = arr.context_timestamp.diff(-1).div(60).abs()
        arr['all_diff_3'] = (arr.context_timestamp - arr.context_timestamp.max()).div(60).abs()
        arr['all_diff_4'] = (arr.context_timestamp - arr.context_timestamp.min()).div(60).abs()
        arr['all_history'] = arr.new_trade.cumsum()
        arr['all_history'] = map(lambda x: x[:-1],arr.history)
        return arr[['instance_id','all_diff_1','all_diff_2','all_diff_3','all_diff_4','all_history']]
    
    arr = data.groupby(['user_id']).apply(help)
    arr = arr.reset_index(drop=True)
    data = pd.merge(data,arr,on=['instance_id'],how = 'left')
    del data['new_trade']
    return data

def first_and_next_item(data,date):
    #data.sort_values(by="context_timestamp" , ascending=True)#升序
    data['new_trade'] = map(lambda x,y:  str(x) if y  < date  else ( str(0) ) , data.is_trade,data.day)
    def help(arr):
        arr = arr.sort_values(by=["context_timestamp"],ascending = True)
        arr['item_diff_1'] = arr.context_timestamp.diff().div(60).abs()
        arr['item_diff_2'] = arr.context_timestamp.diff(-1).div(60).abs()
        arr['item_diff_3'] = (arr.context_timestamp - arr.context_timestamp.max()).div(60).abs()
        arr['item_diff_4'] = (arr.context_timestamp - arr.context_timestamp.min()).div(60).abs()
        arr['item_history'] = arr.new_trade.cumsum()
        arr['item_history'] = map(lambda x: x[:-1],arr.item_history)
        return arr[['instance_id','item_diff_1','item_diff_2','item_diff_3','item_diff_4','item_history']]
    
    arr = data.groupby(['user_id','item_id']).apply(help)
    arr = arr.reset_index(drop=True)
    data = pd.merge(data,arr,on=['instance_id'],how = 'left')
    del data['new_trade']
    return data

def first_and_next_cat1(data,date):
    #data.sort_values(by="context_timestamp" , ascending=True)#升序
    data['new_trade'] = map(lambda x,y:  str(x) if y  < date  else ( str(0) ) , data.is_trade,data.day)
    def help(arr):
        arr = arr.sort_values(by=["context_timestamp"],ascending = True)
        arr['cat1_diff_1'] = arr.context_timestamp.diff().div(60).abs()
        arr['cat1_diff_2'] = arr.context_timestamp.diff(-1).div(60).abs()
        arr['cat1_diff_3'] = (arr.context_timestamp - arr.context_timestamp.max()).div(60).abs()
        arr['cat1_diff_4'] = (arr.context_timestamp - arr.context_timestamp.min()).div(60).abs()
        arr['cat1_history'] = arr.new_trade.cumsum()
        arr['cat1_history'] = map(lambda x: x[:-1],arr.cat1_history)
        return arr[['instance_id','cat1_diff_1','cat1_diff_2','cat1_diff_3','cat1_diff_4','cat1_history']]
    
    arr = data.groupby(['user_id','category_1']).apply(help)
    arr = arr.reset_index(drop=True)
    data = pd.merge(data,arr,on=['instance_id'],how = 'left')
    del data['new_trade']
    return data

def get_whether_cat_trade(data,sub_data):
    #统计该类型的数据用户有没有购买过
        
    cat_1_user = data.groupby(['category_1','user_id'],as_index = False)
    show = cat_1_user['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['category_1','user_id','cat_1_user_show']
    sub_data = pd.merge(sub_data,show,on=['category_1','user_id'],how = 'left')
    
    trade_data = data.loc[data.is_trade == 1]
    trade_data = trade_data[['category_1','user_id','context_timestamp']]
    trade_data.sort_values(by="context_timestamp" , ascending=False)#降序
    trade_data.columns = ['category_1','user_id','trade_time']
    trade_data.drop_duplicates(subset=['category_1','user_id'], keep='first', inplace=True) 
    
    sub_data = pd.merge(sub_data,trade_data,on=['category_1','user_id'],how = 'left')
  
    #sub_data['trade_already'] = map(lambda x,y:  0 if math.isnan(y) else (1 if x > y else 0 ) , sub_data['context_timestamp'],sub_data['trade_time'])
    sub_data['cat_1_trade_no'] = map(lambda x,y:  1 if math.isnan(y) else (0 if x > y else 1 ) , sub_data['context_timestamp'],sub_data['trade_time'])
    del sub_data['trade_time']    
    return sub_data

def get_whether_shop_trade(data,sub_data):
    #统计该类型的数据用户有没有购买过这个商铺的东西     
    shop_user = data.groupby(['shop_id','user_id'],as_index = False)
    show = shop_user['is_trade'].agg([np.size])
    show = show.reset_index()
    show.columns = ['shop_id','user_id','shop_user_show']
    sub_data = pd.merge(sub_data,show,on=['shop_id','user_id'],how = 'left')
    
    trade_data = data.loc[data.is_trade == 1]
    trade_data = trade_data[['shop_id','user_id','context_timestamp']]
    trade_data.sort_values(by="context_timestamp" , ascending=False)#降序
    trade_data.columns = ['shop_id','user_id','trade_time']
    trade_data.drop_duplicates(subset=['shop_id','user_id'], keep='first', inplace=True) 
    
    sub_data = pd.merge(sub_data,trade_data,on=['shop_id','user_id'],how = 'left')
  
    #sub_data['trade_already'] = map(lambda x,y:  0 if math.isnan(y) else (1 if x > y else 0 ) , sub_data['context_timestamp'],sub_data['trade_time'])
    sub_data['shop_trade_no'] = map(lambda x,y:  1 if math.isnan(y) else (0 if x > y else 1 ) , sub_data['context_timestamp'],sub_data['trade_time'])
    del sub_data['trade_time']    
    return sub_data



    
def cal_sum(train,test,online = False):
    data = pd.concat([train,test])
    data['ss'] = map(lambda x:1,data.is_trade)
    
    data['sum'] = data.groupby(['user_id'])['ss'].cumsum()
    data['sum_day'] = data.groupby(['user_id','day'])['ss'].cumsum() #当前用户今天第几次点击
    data['sum_first_day'] = map(lambda x: 1 if x == 1 else 0,data.sum_day)#当前用户今天是否第一次点击
    y = data.sum_day.max()
    data['sum_last_day'] =map(lambda x: 1 if x == y else 0,data.sum_day)
    
    data['sum_item']=data.groupby(['item_id','user_id'])['ss'].cumsum()
    data['sum_item_day'] = data.groupby(['item_id','user_id','day'])['ss'].cumsum() 
    data['sum_item_first_day'] = map(lambda x: 1 if x == 1 else 0,data.sum_item_day)
    y = data.sum_item_day.max()
    data['sum_item_last_day'] =map(lambda x: 1 if x == y else 0,data.sum_item_day)
    
    data['sum_cat1']=data.groupby(['category_1','user_id'])['ss'].cumsum()
    data['sum_cat1_day'] = data.groupby(['category_1','user_id','day'])['ss'].cumsum() 
    data['sum_cat1_first_day'] = map(lambda x: 1 if x == 1 else 0,data.sum_cat1_day)
    y = data.sum_cat1_day.max()
    data['sum_cat1_last_day'] =map(lambda x: 1 if x == y else 0,data.sum_cat1_day)
     
    data['sum_shop']=data.groupby(['shop_id','user_id'])['ss'].cumsum()
    data['sum_shop_day'] = data.groupby(['shop_id','user_id','day'])['ss'].cumsum() 
    data['sum_shop_first_day'] = map(lambda x: 1 if x == 1 else 0,data.sum_shop_day)
    y = data.sum_shop_day.max()
    data['sum_shop_last_day'] =map(lambda x: 1 if x == y else 0,data.sum_shop_day)
    
    del data['ss']
    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    else:
        train = data.loc[data.day < 25]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 25]  # 暂时先使用第24天作为验证集
    return train,test


def gen_feature(online):
     if online == False:
         data = pd.read_csv('./data/round1_ijcai_18_train_20180301.txt', sep=' ') #
         data.drop_duplicates(['instance_id'],inplace = True)   
         
         data = convert_data(data)
         data = convert_cat(data)
         
         start =time.time()
         data = first_and_next(data,date = 24)         
         end = time.time()
         print(end - start)
         
         start =time.time()
         data = first_and_next_2(data,date = 24)
         end = time.time()
         print(end - start)
         
         start =time.time()
         data = first_and_next_item(data,date= 24)
         end = time.time()
         print(end - start)
         
         start =time.time()
         data = first_and_next_cat1(data,date = 24)
         end = time.time()
         print(end - start)
         
         
         data.loc[data.day< 24].to_csv('./data/data_1/offline_train.csv', index=False,sep=' ')
         data.loc[data.day == 24].to_csv('./data/data_1/offline_test.csv', index=False,sep=' ')
     
     else:
         data = pd.read_csv('./data/round1_ijcai_18_train_20180301.txt', sep=' ') #
         data.drop_duplicates(['instance_id'],inplace = True)   
         #instance = set(data.instance_id) - set([892803262387109244, 4773277620195885623, 8860419674042916065])
         #instance = list(instance)
 
         
         test_a = pd.read_csv('./data/round1_ijcai_18_test_a_20180301.txt', sep = ' ') #
         test_b = pd.read_csv('./data/round1_ijcai_18_test_b_20180418.txt', sep = ' ')
         test = pd.concat([test_a,test_b])
         
         instance = set(data.instance_id) & set(test.instance_id)
         instance = set(data.instance_id) - instance
         instance = list(instance)
         data = data[data.instance_id.isin(instance)] 
         
         del test_a,test_b
         gc.collect()
             
         
         data = pd.concat([data,test])
         
         data.is_trade = data.is_trade.fillna(0).astype(int)
         
         data = convert_data(data)
         data = convert_cat(data)
        
         start =time.time()
         data= first_and_next(data,date = 25)
         end = time.time()
         print(end - start)
         
         start =time.time()
         data = first_and_next_2(data,date = 25)
         end = time.time()
         print(end - start)
         
         start =time.time()
         data = first_and_next_item(data,date = 25)
         end = time.time()
         print(end - start)
         
         start =time.time()
         data = first_and_next_cat1(data,date = 25)
         end = time.time()
         print(end - start)
         
         data.loc[data.day < 25].to_csv('./data/data_1/online_train.csv', index=False,sep=' ')
         data.loc[data.day == 25].to_csv('./data/data_1/online_test.csv', index=False,sep=' ')    

if __name__ == "__main__":
    
     online = True# 这里用来标记是 线下验证 还是 在线提交
     #gen_feature(online)

    
     if True:
         if online == False:
           
            train = pd.read_csv('./data/data_1/offline_train.csv', sep=' ', dtype={'history':str,'all_history':str,'cat1_history':str,'item_history':str})
            test = pd.read_csv('./data/data_1/offline_test.csv', sep=' ', dtype={'history':str,'all_history':str,'cat1_history':str,'item_history':str})
            data = pd.concat([train,test])
            data = convert_data(data)
            #data = convert_cat(data)
            data = cross_feature_two_one_day(data)
            data = cross_feature_two_one_hour(data)
            data = cross_feature_two_half_hour(data)
            data = cross_feature_two(data)
            data = convert_predict(data)
            data = convert_pro(data)
            data = count(data)
            train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
            test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
            
         elif online == True:
    
            train = pd.read_csv('./data/data_1/online_train.csv', sep=' ',dtype={'history':str,'all_history':str,'cat1_history':str,'item_history':str})
            test = pd.read_csv('./data/data_1/online_test.csv', sep=' ', dtype={'history':str,'all_history':str,'cat1_history':str,'item_history':str})
            data = pd.concat([train,test])
            data = convert_data(data)
            #data = convert_cat(data)
            data = cross_feature_two_one_day(data)
            data = cross_feature_two_one_hour(data)
            data = cross_feature_two_half_hour(data)
            data = cross_feature_two(data)
            data = convert_predict(data)
            data = convert_pro(data)
            data = count(data)
            train = data.loc[data.day < 25]  # 18,19,20,21,22,23,24
            test = data.loc[data.day == 25] # 暂时先使用第24天作为验证集
         
            
         del data
         gc.collect()
    
         train = get_whether_cat_trade(train,train)
         train = get_whether_shop_trade(train,train)
       
         
         test =get_whether_cat_trade(train,test)
         test = get_whether_shop_trade(train,test)
         
         train,test = cal_sum(train,test,online)
         
         train['history'] = encode_feature(train['history'] )
         train['all_history'] = encode_feature(train['all_history'] )
         train['cat1_history'] = encode_feature(train['cat1_history'] )
         train['item_history'] = encode_feature(train['item_history'] )
         
         test['history'] = encode_feature(test['history'] )
         test['all_history'] = encode_feature(test['all_history'] )
         test['cat1_history'] = encode_feature(test['cat1_history'] )
         test['item_history'] = encode_feature(test['item_history'] )
         
         train['history'] = train['history'].astype(int)
         train['all_history'] = train['all_history'].astype(int)
         train['cat1_history'] = train['cat1_history'].astype(int)
         train['item_history'] = train['item_history'].astype(int)
         
         test['history'] = test['history'].astype(int)
         test['all_history'] = test['all_history'].astype(int)
         test['cat1_history'] = test['cat1_history'].astype(int)
         test['item_history'] = test['item_history'].astype(int)
    
         #所有样本的cat0一样 
         features = list(train.columns)
         no_need = ['instance_id','user_id','context_timestamp','context_id','is_trade','time',
                    'item_category_list', 'item_property_list', 'predict_category_property']
         overfitting = ['item_ctr','user_ctr','shop_ctr','cat_1_ctr'] #'item_count','user_count','shop_count',
         no_need += overfitting
         features = list(set(features) - set(no_need))
        #容易过拟合的特征
    
        
         target = ['is_trade']
        
         xgb_pars = {
                    'eta': 0.05,
                    'gamma': 0,
                    'max_depth': 7,
                    'min_child_weight': 10,
                    'max_delta_step': 0,
                    'subsample': 1,
                    'colsample_bytree': 0.5,
                    'colsample_bylevel': 0.5,
                    'lambda': 1,
                    'alpha': 0,
                    'tree_method': 'auto',
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'nthread': 20,
                    'seed': 42,
                    'silent': 0
                    }
         if online == False:
             Dtrain = xgb.DMatrix(train[features], train[target],feature_names=features)
             Dtest = xgb.DMatrix(test[features], test[target],feature_names=features)
             watchlist = [(Dtrain, 'train'), (Dtest, 'val')]
            
             clf = xgb.train(xgb_pars,Dtrain,num_boost_round=450,verbose_eval=1, evals=watchlist)
            
             #xx = clf.predict(Dtrain, output_margin=False, ntree_limit=0, pred_leaf=True)
      
             train['lgb_predict'] = clf.predict(Dtrain)
             print('train log_loss',log_loss(train[target], train['lgb_predict']))
             
             test['lgb_predict'] = clf.predict(Dtest)
             print('test log_loss',log_loss(test[target], test['lgb_predict']))
             
             """
             xgb_feature = clf.predict(Dtrain,pred_leaf=True)
             xgb_feature = pd.DataFrame(xgb_feature,columns=["xgb_{}".format(i+1) for i in range(xgb_feature.shape[1])])
             xgb_feature['instance_id'] = train['instance_id']
             xgb_feature['is_trade'] = train['is_trade']
             xgb_feature.to_csv('./offline/xgb_train_feature.csv', index=False,sep=' ')
             
             xgb_feature = clf.predict(Dtest,pred_leaf=True)
             xgb_feature = pd.DataFrame(xgb_feature,columns=["xgb_{}".format(i+1) for i in range(xgb_feature.shape[1])])
             xgb_feature['instance_id'] = test['instance_id']
             xgb_feature['is_trade'] = test['is_trade']
             xgb_feature.to_csv('./offline/xgb_test_feature.csv', index=False,sep=' ')
             """

         else:
             Dtrain = xgb.DMatrix(train[features], train[target],feature_names=features)
             Dtest = xgb.DMatrix(test[features],feature_names=features)
             watchlist = [(Dtrain, 'train')]
             clf = xgb.train(xgb_pars,Dtrain,num_boost_round=450,verbose_eval=1,evals=watchlist)
            
             #xx = clf.predict(Dtrain, output_margin=False, ntree_limit=0, pred_leaf=True)
    
             train['predicted_score'] = clf.predict(Dtrain)
             print('train log_loss',log_loss(train[target], train['predicted_score']))        
             test['predicted_score'] = clf.predict(Dtest)    
             
             train[['instance_id','is_trade', 'predicted_score']].to_csv('new_xgb_train_baseline.csv', index=False,sep=' ')
             
             test_b = pd.read_csv('./data/round1_ijcai_18_test_b_20180418.txt', sep = ' ')
             
             #test_b = test_b['instance_id']
             test_b = pd.merge(test_b,test[['instance_id','predicted_score']],on =['instance_id'],how = 'left')
             
             test_b[['instance_id', 'predicted_score']].to_csv('new_xgb_test_baseline.csv', index=False,sep=' ')


             """
             xgb_feature = clf.predict(Dtrain,pred_leaf=True)
             xgb_feature = pd.DataFrame(xgb_feature,columns=["xgb_{}".format(i+1) for i in range(xgb_feature.shape[1])])
             xgb_feature['instance_id'] = train['instance_id']
             xgb_feature['is_trade'] = train['is_trade']
             xgb_feature.to_csv('./online/xgb_train_feature.csv', index=False,sep=' ')
             
             xgb_feature = clf.predict(Dtest,pred_leaf=True)
             xgb_feature = pd.DataFrame(xgb_feature,columns=["xgb_{}".format(i+1) for i in range(xgb_feature.shape[1])])
             xgb_feature['instance_id'] = test['instance_id']
             #xgb_feature['is_trade'] = test['is_trade']
             xgb_feature.to_csv('./online/xgb_test_feature.csv', index=False,sep=' ')
             """
# =============================================================================
#offline = 450
#('train log_loss', 0.07457367394871987)
#('test log_loss', 0.07922329891091333)

#online  logloss,0.08474
#=============================================================================
