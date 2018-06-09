#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:20:30 2018

@author: flyaway
"""

list_feature = ['most_proerty','most_predict_cat','most_predict_pro']

no_list_feature = [ 'item_id','item_brand_id', 'item_city_id',
       'item_price_level', 'item_sales_level', 'item_collected_level',
       'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level',
       'user_occupation_id', 'user_star_level',
       'context_page_id', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate',
       'shop_star_level', 'shop_score_service', 'shop_score_delivery',
       'shop_score_description','is_week','hour','category_1'] 

item_price_level: 14
item_sales_level: 18
item_collected_level: 18
item_pv_level: 22
user_gender_id: 3
user_age_level: 9
user_occupation_id: 5
user_star_level: 12
context_page_id: 20
shop_review_num_level: 25
shop_star_level: 22

shop_review_positive_rate: 11825 可以考虑分桶,也可以考虑直接用数值型特征，而且都已经归一化了，这个大部分都是1,参考价值不大
shop_score_service: 16362 可以考虑分桶
shop_score_delivery: 16423 可以考虑分桶
shop_score_description: 16463 可以考虑分桶

考虑使用每一天的转化率做矫正
data.loc[data.day == 18].shape (78261, 36)
data.loc[data.day == 19].shape (70926, 36)
data.loc[data.day == 20].shape (68377, 36)
data.loc[data.day == 21].shape (71193, 36)
data.loc[data.day == 22].shape (68309, 36)
data.loc[data.day == 23].shape (63610, 36)
data.loc[data.day == 24].shape (57411, 36)
data.loc[data.day == 25].shape (18371, 26)

计算每天的转化率，根据转化率每天都在下降的趋势，将最后的结果减去一个值
data = 18,78261,1560,0.0199
data = 19,70926,1392,0.0196
data = 20,68377,1321,0.0193
data = 21,71193,1371,0.01926
data = 22,68309,1286,0.018826
data = 23,63610,1096,0.01723
data = 24,57411,968,0.01686

考虑采样24号的数据做线下的验证，因为使用的first_and_next()依靠每一天的数据,但是25号的给的数据经过采样，数据明显变少，
会导致计算有较大偏差，因为25号的数据大部分用户只有一条样本，没办法计算
a.距离上一次和下一次点击广告的时间差 
b.距离当日第一次和最后一次点击广告的时间
c.问题：由于测试那天样本被降采样了，所以测试集上计算这两个值会有偏差

训练集和验证集user_id重合样本有6718　占验证集11.70%
训练集和测试集user_id重合样本有3626  占测试集19.74%

所以是否考虑计算去除天的影响，直接计算，或者以聊天为单位
距离用户上一次和下一次一次点击广告的时间差
距离用户第一次和最后一次点击广告的时间差



其他人同类型题目，特征的考虑
1.利用历史流水 
用户最近一次安装App时间差 
用户最近一次安装同类App时间差 
用户同类别广告浏览统计 
用户最近几天安装App统计 
用户历史浏览position hash 
用户历史浏览App hash 
同一个position处用户浏览不同素材个数 
…… 
2.label窗特征 
当日数据-> 
距离上一次和下一次点击广告的时间差 
距离当日第一次和最后一次点击广告的时间 
当日点击广告的持续时间 
短时间内重复点击广告的计数以及次序 
3.CVR特征 
4.转化率的贝叶斯平滑，看论文:Click-Through Rate Estimation for Rare Events In Online Adversting 
由于数据稀疏性的原因，直接观测到的CVR与真实的CVR之间 
的误差较大。因此利用贝叶斯平滑对CVR预估进行优化 
·对于某广告,C表示回流次数，I表示点击次数 
·用平滑转化率r作为特征 
5.使用01串表示用户历史点击----->offline提取的数据测试可以，online提取的数据就过拟合，不知道哪里有问题
6.均值调整 
7.数据量过大,可以用流式统计方法,避免整体读入内存中 
8.借鉴以前three idiots的方案 
9.线上/线下验证集的挑选 
先利用后几天的来验证下所用天的label的情况,他这里把25号和30号都去掉了(也有队伍保留了30号的数据,但是删除了转化率突变的app数据)
10.特殊的Leak特征  
11.利用xgboost训练出来的决策树,对连续特征分箱
12.数据处理时的经验 
特征提取使用Python的numpy、 pandas及Map Reduce 
Tips 
• 使用shell脚本并行提取特征 
• 使用python的multiprocessing库可以加速特征的提取 
• 使用numpy.savez及scipy.csr_matrix完成特征文件的持久化 
13.定义了app非当天转化率来对数据做清洗 
训练数据的末尾几天由于转化时间的滞后性，存在错误标签的样本，越靠近第30天，错误样本的比例越大 
14.时间特征 
一天24小时分成48个半小时，点击事件发生区间作为特征 
15.PNN和NFFM 
