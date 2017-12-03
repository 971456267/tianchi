#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:39:06 2017

@author: flyaway
"""
import pandas as pd
import numpy as np
import os
import gc
import time as T
import sys
#http://blog.csdn.net/u013401853/article/details/73368850
from math import sin,asin,cos,radians,fabs,sqrt
from sklearn import preprocessing
EARTH_RADIUS = 6371
def encode_feature(values):
    uniq = values.unique()
    mapping = dict(zip(uniq, range(1, len(uniq) + 1)))
    return values.map(mapping)

def hav(theta):
    s = np.sin(theta / 2.0)
    return s * s

def get_distance_hav(lat0,lng0,lat1,lng1):
    #经纬度转化成弧度
    lat0 = np.radians(lat0)
    lat1 = np.radians(lat1)
    lng0 = np.radians(lng0)
    lng1 = np.radians(lng1)
    
    dlng = np.fabs(lng0 - lng1)
    dlat = np.fabs(lat0 - lat1)
    
    h = hav(dlat) + np.cos(lat0)*np.cos(lat1)*hav(dlng)
    
    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))
    
    return distance




def get_feature(train,test,mall_name):
    #train = pd.read_csv(train_path)
    user_latitude = np.asarray(train['user_latitude'])
    user_longitude = np.asarray(train['user_longitude'])
    shop_latitude = np.asarray(train['shop_latitude'])
    shop_longitude = np.asarray(train['shop_longitude'])
    dist = get_distance_hav(user_latitude,user_longitude,shop_latitude,shop_longitude)
    train['distance'] = dist
    #train = train.sample(frac = 1.0)
    
    #基于距离删除一部分样本
    #train_neg_sample = train_neg.assign(rn=train_neg.sort_values(['distance'], ascending=True)
    #                                  .groupby('row_id').cumcount()+1).query('rn < 10')
    #del train_neg_sample['rn']
    
    #train = pd.concat([train_pos,train_neg_sample],axis = 0)
    
    train_pos = train[train.label == 1]
    wifi_info = train_pos['wifi_infos']
    wifi_info = np.asarray(wifi_info)
    del train_pos
    gc.collect()
    
    wifi = dict()
    index = 1
    for i in xrange(wifi_info.shape[0]):
        temp = wifi_info[i].split(";")
        for item in temp:
            t = item.split("|")[0]
            if t in wifi:
                wifi[t] += 1
            else:
                wifi[t] = 1
   
    del_wifi = []
    for item in wifi:
        if wifi[item] < 50:
            del_wifi.append(item)
    
    val_wifi = wifi.keys()
    val_wifi = set(val_wifi) - set(del_wifi)
    
    wifi_id= {}
    wifi_max = -sys.maxsize
    wifi_min = sys.maxsize
    for i in xrange(wifi_info.shape[0]):
        temp = wifi_info[i].split(";")
        for item in temp:
            t = item.split('|')[0]
            val = abs(int(item.split('|')[1]))
            wifi_max = max(wifi_max,val)
            wifi_min = min(wifi_min,val)
            if t in val_wifi and t not in wifi_id:
                wifi_id[t] = index
                index += 1
            else:
                continue
                

    train = train.sample(frac = 1)    
    
    #test = pd.read_csv(test_path)
    #print test.shape
    user_latitude = np.asarray(test['user_latitude'])
    user_longitude = np.asarray(test['user_longitude'])
    shop_latitude = np.asarray(test['shop_latitude'])
    shop_longitude = np.asarray(test['shop_longitude'])
    dist = get_distance_hav(user_latitude,user_longitude,shop_latitude,shop_longitude)
    
    test['distance'] = dist
    #基于距离删除一部分样本
    #test = test.assign(rn=test.sort_values(['distance'], ascending=True)
    #                                  .groupby('row_id').cumcount()+1).query('rn <= 20')
    #del test['rn']
    
    test = test.sample(frac = 1)
    
    test['label'] = 2 *np.ones(test.shape[0],int)


    
    cols = list(test)
    train = train.loc[:,cols]
    
    data = pd.concat([train,test],axis = 0)
    train_shape  = train.shape
    
    #求出训练集中的wifi_id

    print(mall_name,len(wifi_id),train.shape,test.shape)    
    del train,test
    gc.collect()
    
    #开始提取特征
    #处理成libsvm格式的数据
    #label 放在第０位
    label = data['label'].astype(int)
    label = np.asarray(label).reshape((label.shape[0],1))
    
    #price为1号特征，按长度等分
    price_feature = data['price'].astype(int)#归一化price
    price_feature = price_feature.reshape((price_feature.shape[0],1))
    price_feature = preprocessing.scale(price_feature)

    price_buck1 = pd.cut(data['price'],5,labels = False)
    price_buck2 = pd.qcut(data['price'],4,labels=False) #label=false即可值获取分位数的编号
    price_buck1 = price_buck1.reshape((price_buck1.shape[0],1))
    price_buck2 = price_buck2.reshape((price_buck2.shape[0],1))
    price_buck1_max = 4
    price_buck2_max = 3
    """
    def get_stats(group):
        return {'min':group.min(),'max':group.max(),'count':group.count(),'mean':group.mean()}
    grouped=data.price.groupby(price_buck)
    grouped.apply(get_stats).unstack()
    """
   


    #category为2号特征
    print '5'
    cat_feature = pd.factorize(data['category_id'].values , sort=True)[0] + 1  
    cat_feature = cat_feature.reshape((cat_feature.shape[0],1))
    cat_max = np.amax(cat_feature)
    
    #user为3号特征
    #user_feature = pd.factorize(data['user_id'].values , sort=True)[0] + 1  
    #user_feature = np.asarray(user_feature).reshape((user_feature.shape[0],1))
    #user_max = np.amax(user_feature) 
    #distance为4号特征
    distance_feature = np.asarray(data['distance'])
    distance_feature = distance_feature.reshape((distance_feature.shape[0],1))
    distance_feature = preprocessing.scale(distance_feature)
    
    #shop_count为5号特征
    count_feature = np.asarray(data['shop_count'])
    count_feature = distance_feature.reshape((distance_feature.shape[0],1))
    count_feature = preprocessing.scale(count_feature)  #这里可以考虑分桶，看一下怎么实现pd.cut 和pd.pcut,分桶之后就用onehot
    
    count_buck1 = pd.cut(data['shop_count'],10,labels = False)
    count_buck2 = pd.qcut(data['shop_count'],5,labels=False) #label=false即可值获取分位数的编号
    count_buck1 = price_buck1.reshape((price_buck1.shape[0],1))
    count_buck2 = price_buck2.reshape((price_buck2.shape[0],1))
    count_buck1_max = 9
    count_buck2_max = 4
    
    time = data['time_stamp']
    time = np.array(time)
    hour = []
    week = []
    for i in xrange(len(time)):
        s = int(time[i][11:13])
        if s >= 10 & s <= 22:
            hour.append(s)
        else:
            hour.append(0)
        week.append(T.strptime(time[i][0:10],'%Y-%m-%d').tm_wday + 1)
    
    #hour为6号特征
    hour = np.array(hour)
    hour = hour.reshape(hour.shape[0],1)
    
    #week为7号特征
    week = np.array(week)
    week = week.reshape(week.shape[0],1)
    
    #处理row_id，由于手动生成了负样本，所以一个row_id表示了一次付款的正样本和负样本集合,做特征是无意义的，但是需要row_id 做标识
    
    df = data.loc[:,['row_id','label','shop_id']] 
    df.to_csv("../shop_id/" + mall_name + '.csv',index = False)
    
    #以libsvm格式写数据 注意切分train,test
    wifi_info = data['wifi_infos']
    wifi_info = np.asarray(wifi_info)
    #训练集特征
    wf = open("../train_sample/" + mall_name + '.txt','w')
    
    for i in xrange(train_shape[0]):
        tmp = dict()
        #tmp[0] = label[i][0]
        tmp[0] = [1,price_feature[i][0]]
        tmp[1] = [1,distance_feature[i][0]]
        tmp[2] = [1,count_feature[i][0]]
        tmp[3] = [hour[i][0],1] #hour[i][0]
        tmp[4] = [week[i][0],1] #week[i][0]
        tmp[5] =  [[cat_feature[i][0],1],[cat_max, 1 if cat_feature[i][0] == cat_max else 0 ]]
        #tmp[6] =  [[user_feature[i][0],1],[user_max, 1 if user_feature[i][0] == user_max else 0]]
        tmp[6] = {}
        
        temp = wifi_info[i].split(";")
        
        for item in temp:
            t = item.split('|')[0]
            if t in wifi_id:
                val = abs(int(item.split('|')[1]))
                tmp[6][wifi_id[t]] = ( val - wifi_min)*1.0 / (wifi_max - wifi_min)
            else:
                continue
        
        tmp[7] = [[price_buck1[i][0],1],[price_buck1_max,1 if price_buck1[i][0] == price_buck1_max else 0 ]]
        tmp[8] = [[price_buck2[i][0],1],[price_buck2_max,1 if price_buck2[i][0] == price_buck2_max else 0 ]]
        tmp[9] = [[count_buck1[i][0],1],[count_buck1_max,1 if count_buck1[i][0] == count_buck1_max else 0 ]]
        tmp[10] = [[count_buck2[i][0],1],[count_buck2_max,1 if count_buck2[i][0] == count_buck2_max else 0 ]]
        
        wf.write("%s "%(label[i][0]))
        wf.write("%s:%s:%s " %(0,tmp[0][0],tmp[0][1]))
        wf.write("%s:%s:%s " %(1,tmp[1][0],tmp[1][1]))
        wf.write("%s:%s:%s " %(2,tmp[2][0],tmp[2][1]))
        wf.write("%s:%s:%s " %(3,tmp[3][0],tmp[3][1]))
        wf.write("%s:%s:%s " %(4,tmp[4][0],tmp[4][1]))
        wf.write("%s:%s:%s " %(5,tmp[5][0][0],tmp[5][0][1]))
        wf.write("%s:%s:%s " %(5,tmp[5][1][0],tmp[5][1][1]))
        #wf.write("%s:%s:%s " %(6,tmp[6][0][0],tmp[6][0][1]))
        #wf.write("%s:%s:%s " %(6,tmp[6][1][0],tmp[6][1][1]))
        
        for item in tmp[6]:
            wf.write("%s:%s:%s " %(6,item,tmp[6][item]))
            
        wf.write("%s:%s:%s " %(7,tmp[7][0][0],tmp[7][0][1]))
        wf.write("%s:%s:%s " %(7,tmp[7][1][0],tmp[7][1][1]))
        wf.write("%s:%s:%s " %(8,tmp[8][0][0],tmp[8][0][1]))
        wf.write("%s:%s:%s " %(8,tmp[8][1][0],tmp[8][1][1]))
        wf.write("%s:%s:%s " %(9,tmp[9][0][0],tmp[9][0][1]))
        wf.write("%s:%s:%s " %(9,tmp[9][1][0],tmp[9][1][1]))
        
        wf.write("%s:%s:%s " %(10,tmp[10][0][0],tmp[10][0][1]))
        wf.write("%s:%s:%s " %(10,tmp[10][1][0],tmp[10][1][1]))

        wf.write("\n")
    wf.close()
    
    #测试集特征
    wf = open("../test_sample/" + mall_name + '.txt','w')
    for i in xrange(train_shape[0],label.shape[0]):
         #tmp[0] = label[i][0]
        tmp[0] = [1,price_feature[i][0]]
        tmp[1] = [1,distance_feature[i][0]]
        tmp[2] = [1,count_feature[i][0]]
        tmp[3] = [hour[i][0],1] #hour[i][0]
        tmp[4] = [week[i][0],1] #week[i][0]
        tmp[5] =  [[cat_feature[i][0],1],[cat_max, 1 if cat_feature[i][0] == cat_max else 0 ]]
        #tmp[6] =  [[user_feature[i][0],1],[user_max, 1 if user_feature[i][0] == user_max else 0]]
        tmp[6] = {}
        
        temp = wifi_info[i].split(";")
        
        for item in temp:
            t = item.split('|')[0]
            if t in wifi_id:
                 val = abs(int(item.split('|')[1]))
                 tmp[6][wifi_id[t]] = ( val - wifi_min)*1.0 / (wifi_max - wifi_min)
            else:
                continue
        
        tmp[7] = [[price_buck1[i][0],1],[price_buck1_max,1 if price_buck1[i][0] == price_buck1_max else 0 ]]
        tmp[8] = [[price_buck2[i][0],1],[price_buck2_max,1 if price_buck2[i][0] == price_buck2_max else 0 ]]
        tmp[9] = [[count_buck1[i][0],1],[count_buck1_max,1 if count_buck1[i][0] == count_buck1_max else 0 ]]
        tmp[10] = [[count_buck2[i][0],1],[count_buck2_max,1 if count_buck2[i][0] == count_buck2_max else 0 ]]
        
        wf.write("%s "%(label[i][0]))
        wf.write("%s:%s:%s " %(0,tmp[0][0],tmp[0][1]))
        wf.write("%s:%s:%s " %(1,tmp[1][0],tmp[1][1]))
        wf.write("%s:%s:%s " %(2,tmp[2][0],tmp[2][1]))
        wf.write("%s:%s:%s " %(3,tmp[3][0],tmp[3][1]))
        wf.write("%s:%s:%s " %(4,tmp[4][0],tmp[4][1]))
        wf.write("%s:%s:%s " %(5,tmp[5][0][0],tmp[5][0][1]))
        wf.write("%s:%s:%s " %(5,tmp[5][1][0],tmp[5][1][1]))
        #wf.write("%s:%s:%s " %(6,tmp[6][0][0],tmp[6][0][1]))
        #wf.write("%s:%s:%s " %(6,tmp[6][1][0],tmp[6][1][1]))
        for item in tmp[6]:
            wf.write("%s:%s:%s " %(6,item,tmp[6][item]))
      
        wf.write("%s:%s:%s " %(7,tmp[7][0][0],tmp[7][0][1]))
        wf.write("%s:%s:%s " %(7,tmp[7][1][0],tmp[7][1][1]))
        wf.write("%s:%s:%s " %(8,tmp[8][0][0],tmp[8][0][1]))
        wf.write("%s:%s:%s " %(8,tmp[8][1][0],tmp[8][1][1]))
        wf.write("%s:%s:%s " %(9,tmp[9][0][0],tmp[9][0][1]))
        wf.write("%s:%s:%s " %(9,tmp[9][1][0],tmp[9][1][1]))
        
        wf.write("%s:%s:%s " %(10,tmp[10][0][0],tmp[10][0][1]))
        wf.write("%s:%s:%s " %(10,tmp[10][1][0],tmp[10][1][1]))
       
        wf.write("\n")
    
    wf.close()

"""  
if __name__ == '__main__':
    files = os.listdir("./train_sample/")
    for f in files:
        start = T.time()
        mall_name = f[0:-4]
        train_path = "./train_sample/" + f
        test_path = "./test_sample/" + f
        get_feature(train_path,test_path,mall_name)
        end = T.time()
        print('cost_time',end-start)
"""