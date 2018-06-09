#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:00:46 2018

@author: flyaway
"""
import pandas as pd
import collections,time
from collections import Counter
import numpy as np
import hashlib,csv

def hashstr(str, nr_bins = int(1e+6)):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_hashed_fm_feats(feats, nr_bins = int(1e+6)):
    feats = ['{0}:{1}:1'.format(field, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

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

def convert_time(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['is_week'] = data.time.apply(get_week)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    """
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])
    """
    return data

def convert_cat(data):
    #所有商品的一级类目都一样，二级类目为13个，三级类目2,还有一个为-1
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else "-1")
        data['category_%d'%(i)] = data['category_%d'%(i)].astype(int)
    del data['item_category_list']
    return data

def convert_pro(data):
    #for all 
    property_ = []
    for item in data['item_property_list']:
        property_.extend(item.split(";"))
    
    most_count = Counter(property_).most_common(1000) #1000,出现在前1000的属性 
    property_ = []
    for item in most_count:
        property_.append(item[0])  
    
    property_set = set(property_)
    
    
    property_dict = collections.defaultdict(lambda : -1)
    for i,item in enumerate(property_):
        property_dict[item] = i #给属性编号  
    most_proerty = []
    for item in data['item_property_list']:
        tmp = set(item.split(";"))
        tmp = tmp & property_set
        sub = []
        tmp = list(tmp)
        
        sub = [property_dict[x] for x in tmp]
        
        most_proerty.append(sub)
        #most_proerty.append(";".join(list(tmp)))
    data['most_proerty'] = most_proerty
    return data
 
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
        
    most_predict_cat = []
    most_predict_pro = []
    for item in data['predict_category_property']:
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
            
        sub_cat = list(set(sub_cat) & (predict_cat_set))
        sub_pro = list(set(sub_pro) & (predict_pro_set))
        
        
        
        sub_cat = [predict_cat_dict[x] for x in sub_cat]
        sub_pro = [predict_pro_dict[x] for x in sub_pro]
        most_predict_cat.append(sub_cat)
        most_predict_pro.append(sub_pro)
    
    data['most_predict_cat'] = most_predict_cat
    data['most_predict_pro'] = most_predict_pro
    
    return data



def convert_val_to_buck(data,val_feature):
    for feat in val_feature:
        data[feat+'_buck'] = pd.cut(data[feat],10,labels=False)   
        data[feat+'_buck'] = data[feat+'_buck'].astype(str)
        data[feat+'_buck'] = encode_feature(data[feat+'_buck'])
    return data
    
def convert_no_list_feature(data,no_list_feature):
    for feat in no_list_feature:
        data[feat] = encode_feature(data[feat])
    return data


#统计频繁特征
def count(data,no_list_feature,list_feature):
    counts = collections.defaultdict(lambda : [0, 0, 0])
    #csv.register_dialect('mydialect',delimiter=' ', quoting=csv.QUOTE_ALL)  
    #for i,row in enumerate(csv.DictReader(open(args['csv_path']),dialect='mydialect'),start=1):
    for i in range(data.shape[0]):
        row = data.iloc[i,:]
        label = row['is_trade']
        
        for cat_name in no_list_feature:
            value = row[cat_name]
            if label == 0:
                counts[cat_name + ' '+str(value)][0] += 1
            else:
                counts[cat_name + ' '+str(value)][1] += 1
            counts[cat_name + ' '+str(value)][2] += 1
        
        for cat_name in list_feature:
            value = row[cat_name]
            if label != 1:
                for it in value:
                    counts[cat_name + ' ' + str(it)][0] += 1
            else:
                for it in value:
                    counts[cat_name + ' ' + str(it)][1] += 1
            for it in value:
                counts[cat_name + ' '+str(it)][2] += 1
    
    f = open("./count_frquent_feature.csv",'w')
    f.write('Field Value Neg Pos Total Ratio' + '\n')
    for key,(neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
        if total < 10:
            continue
        ratio = round(float(pos)/total,5)
       
        f.write(key+' '+str(neg)+' '+str(pos)+' '+str(total)+' '+str(ratio) + '\n')
    f.close()
            


    
def convert_ffm_format(data,name,feature_dict,feature_name,
                       no_list_feature,list_feature,xgb_feature,online = False):
    if online == False:
        file_name = "./offline/"
    else:
        file_name = "./online/"
    wf = open(file_name + name + '.txt','w')
    
    """   
    for i in range(data.shape[0]):

        item = data.iloc[i,:]  
        feats = []
        for feat in feature_name:
            if feat in no_list_feature or feat in xgb_feature:
                feats.append((feature_dict[feat],str(0)))
            elif feat in list_feature:
                for it in item[feat]:
                    feats.append((feature_dict[feat],str(it))) 
            
 
        
        feats = gen_hashed_fm_feats(feats)
        wf.write(str(item['is_trade']) + ' ' + ' '.join(feats) + '\n')   
         
    """  
    for i in range(data.shape[0]):
        item = data.iloc[i,:]
        #if item['is_trade'] != np.nan:
        wf.write("%s "%(item['is_trade']))
        for feat in feature_name:
            if feat in no_list_feature :
                wf.write("%s:%s:%s " %(feature_dict[feat],item[feat],1))
            elif feat in xgb_feature:
                wf.write("%s:%s:%s " %(feature_dict[feat],item[feat],1))
            elif feat in list_feature:
                for it in item[feat]:
                    wf.write("%s:%s:%s " %(feature_dict[feat],it,1))     
            else:
                print "error"
        wf.write("\n")
    wf.close()

def read_freqent_feats(threshold=10):
    frequent_feats = set()
    high_ratio_feat = []
    csv.register_dialect('mydialect',delimiter=' ', quoting=csv.QUOTE_ALL) 
    for row in csv.DictReader(open('count_frquent_feature.csv'),dialect='mydialect'):
        if float(row['Total']) < threshold:
            continue
        frequent_feats.add(row['Field']+'-'+row['Value'])
        high_ratio_feat.append((row['Field']+'-'+row['Value'],float(row['Ratio'])))
    
    high_ratio_feat = sorted(high_ratio_feat, key=lambda it: it[1])
    high_ratio_feat = [it[0] for it in high_ratio_feat]
    high_ratio_feat = high_ratio_feat[-101:-1]
    return frequent_feats,high_ratio_feat
    

      
if __name__ == "__main__":
    online = False   
    data = pd.read_csv('./data/round1_ijcai_18_train_20180301.txt', sep=' ')
    data.shop_review_positive_rate.replace(-1,0,inplace = True)
    data.drop_duplicates(['instance_id'],inplace = True)
    list_feature = ['most_proerty','most_predict_cat','most_predict_pro']
    #list_feature = []
    val_feature = ['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']
    
    no_list_feature = [ 'item_id','item_brand_id', 'item_city_id','item_price_level', 'item_sales_level', 'item_collected_level','item_pv_level', 
                        'user_id', 'user_gender_id', 'user_age_level','user_occupation_id', 'user_star_level',
                        'context_page_id', 'is_week','hour',
                        'shop_id', 'shop_review_num_level','shop_star_level','category_1','shop_review_positive_rate_buck','shop_score_service_buck','shop_score_delivery_buck', 'shop_score_description_buck']    #context_id,'instance_id','context_timestamp'
    
    if online == False:

        
        data = convert_time(data)
        data = convert_cat(data)
        data = convert_pro(data)
        data = convert_predict(data)
        data = convert_val_to_buck(data,val_feature)
        data = convert_no_list_feature(data,no_list_feature)
        
        #count(data,no_list_feature,list_feature)
        
        #requent_feats,high_ratio_feature = read_freqent_feats(1000000)
        
        train_xgb_feature = pd.read_csv('./offline/xgb_train_feature.csv',sep = ' ')
        del train_xgb_feature['is_trade']
        test_xgb_feature = pd.read_csv('./offline/xgb_test_feature.csv',sep = ' ')
        del test_xgb_feature['is_trade']
        xgb_feature = list(set(train_xgb_feature.columns) - set(['instance_id']))
        
        train_xgb_feature = convert_no_list_feature(train_xgb_feature,xgb_feature)
        test_xgb_feature = convert_no_list_feature(test_xgb_feature,xgb_feature)
        
        feature_name = list_feature + no_list_feature + xgb_feature
        
        feature_dict = collections.defaultdict(lambda : -1)
        for i,item in enumerate(feature_name):
            feature_dict[item] = i

        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        train = pd.merge(train,train_xgb_feature,on = ['instance_id'],how = 'left')
      
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
        test = pd.merge(test,test_xgb_feature,on = ['instance_id'],how = 'left')
  
        start = time.time()
        convert_ffm_format(train,'tr',feature_dict,feature_name,no_list_feature,list_feature,xgb_feature,online)
        end = time.time()
        
        print("gen_feat",end -start)
        convert_ffm_format(test,'va',feature_dict,feature_name,no_list_feature,list_feature,xgb_feature,online)
    else:
        test = pd.read_csv('./data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test.shop_review_positive_rate.replace(-1,0,inplace = True)
        
        data = pd.concat([data,test])
        
        data.is_trade = data.is_trade.fillna(0).astype(int)
        
        data = convert_time(data)
        data = convert_cat(data)
        data = convert_pro(data)
        data = convert_predict(data)
        data = convert_val_to_buck(data,val_feature)
        data = convert_no_list_feature(data,no_list_feature)
        
        count(data,no_list_feature,list_feature)
        
        frequent_feats,high_ratio_feature = read_freqent_feats(10)
        
       
        
        train_xgb_feature = pd.read_csv('./online/xgb_train_feature.csv',sep = ' ')
        del train_xgb_feature['is_trade']
        test_xgb_feature = pd.read_csv('./online/xgb_test_feature.csv',sep = ' ')
        
        xgb_feature = list(set(train_xgb_feature.columns) - set(['instanced_id']))
        
        train_xgb_feature = convert_no_list_feature(train_xgb_feature,xgb_feature)
        test_xgb_feature = convert_no_list_feature(test_xgb_feature,xgb_feature)
        
        feature_name = list_feature + no_list_feature + val_feature + list(frequent_feats) + list(high_ratio_feature) + xgb_feature
        
        feature_dict = collections.defaultdict(lambda : -1)
        for i,item in enumerate(feature_name):
            feature_dict[item] = i

        train = data.loc[data.day < 25]  # 18,19,20,21,22,23,24
        train = pd.merge(train,train_xgb_feature,on = ['instance_id'],how = 'left')
        test = data.loc[data.day == 25]  # 暂时先使用第24天作为验证集
        test = pd.merge(test,test_xgb_feature,on = ['instance_id'],how = 'left')
    
        start = time.time()
        convert_ffm_format(train,'tr',feature_dict,feature_name,no_list_feature,list_feature,val_feature,high_ratio_feature,frequent_feats,xgb_feature,online)
        end = time.time()
        print("gen_feat",end -start)
        convert_ffm_format(test,'ve',feature_dict,feature_name,no_list_feature,list_feature,val_feature,high_ratio_feature,frequent_feats,xgb_feature,online)