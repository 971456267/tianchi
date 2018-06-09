#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

import pickle

xgb_train = pd.read_csv('./new_xgb_train_baseline.csv', sep=' ')
xgb_test = pd.read_csv('./new_xgb_test_baseline.csv', sep=' ')

lgb_train = pd.read_csv('./new_lgb_train_baseline.csv', sep=' ')
lgb_test = pd.read_csv('./new_lgb_test_baseline.csv', sep=' ')

#miao_train = pickle.load(open("./miao/test_offline.pkl",'rb'))
miao_test = pd.read_csv('./miao/result.txt', sep=' ')


print('train log_loss',log_loss(xgb_train['is_trade'],xgb_train['predicted_score']))
print('train log_loss',log_loss(lgb_train['is_trade'],lgb_train['predicted_score']))

xgb_train['predict_blend'] = 0.6 * xgb_train['predicted_score'] + 0.4 * lgb_train['predicted_score']
print('blend log_loss',log_loss(xgb_train['is_trade'],xgb_train['predict_blend']))


xgb_test['predict_blend'] = 0.8 * xgb_test['predicted_score'] + \
                            0.1 * lgb_test['predicted_score'] + \
                            0.1 * miao_test['predicted_score']

result = xgb_test[['instance_id', 'predict_blend']]
result.columns = ['instance_id','predicted_score']
result.to_csv('blend_test_baseline.csv', index=False,sep=' ')
