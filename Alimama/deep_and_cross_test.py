#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:14:49 2018

@author: flyaway
"""
import numpy as np
import pandas as pd
import keras.backend as K
from keras import layers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Input, Embedding, Reshape, Add
from keras.layers import Flatten, merge, Lambda
from keras.models import Model
#from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import time
from sklearn.cross_validation import train_test_split
import gc
from sklearn.metrics import log_loss
# similar to https://github.com/jrzaurin/Wide-and-Deep-Keras/blob/master/wide_and_deep_keras.py


def encode_feature(values):
    uniq = values.unique()
    mapping = dict(zip(uniq,range(1,len(uniq) + 1)))
    return values.map(mapping)

def feature_generate(data):
    data, label, cate_columns, cont_columns = process_data(data)
    embeddings_tensors = []
    continuous_tensors = []
    for ec in cate_columns:
        layer_name = ec + '_inp'
        # For categorical features, we em-bed the features in dense vectors of dimension 6×(category cardinality)**(1/4)
        embed_dim = data[ec].nunique() if int(20 * np.power(data[ec].nunique(), 1/4)) > data[ec].nunique() \
            else int(20 * np.power(data[ec].nunique(), 1/4))
        
        t_inp, t_build = embedding_input(layer_name, data[ec].nunique(), embed_dim)
        embeddings_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    for cc in cont_columns:
        layer_name = cc + '_in'
        t_inp, t_build = continous_input(layer_name)
        continuous_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    inp_layer =  [et[0] for et in embeddings_tensors]
    inp_layer += [ct[0] for ct in continuous_tensors]
    inp_embed =  [et[1] for et in embeddings_tensors]
    inp_embed += [ct[1] for ct in continuous_tensors]
    return data, label, inp_layer, inp_embed,cate_columns, cont_columns

def embedding_input(name, n_in, n_out):
    inp = Input(shape = (1, ), dtype = 'int64', name = name)
    return inp, Embedding(n_in, n_out, input_length = 1)(inp)

def continous_input(name):
    inp = Input(shape=(1, ), dtype = 'float32', name = name)
    return inp, Reshape((1, 1))(inp)



# The optimal hyperparameter settings were 8 cross layers of size 54 and 6 deep layers of size 292 for DCN
# Embed "Soil_Type" column (embedding dim == 15), we have 8 cross layers of size 29   
def create_model(inp_layer, inp_embed):
    #inp_layer, inp_embed = feature_generate(X, cate_columns, cont_columns)
    input = merge(inp_embed, mode = 'concat')
    print(input.shape)
    # deep layer
    for i in range(6):
        if i == 0:
            deep = Dense(272, activation='relu')(Flatten()(input))
        else:
            deep = Dense(272, activation='relu')(deep)

    # cross layer
    cross = CrossLayer(output_dim = input.shape[2].value, num_layer = 8, name = "cross_layer")(input)

    #concat both layers
    output = merge([deep, cross], mode = 'concat')
    output = Dense(1, activation = 'sigmoid')(output)
    model = Model(inp_layer, output) 
    return model

def fit(model,inp_layer, inp_embed, X, y,train_feature,*params): #X_val,y_val
    """
    input = merge(inp_embed, mode = 'concat')
    print(input.shape)
    # deep layer
    for i in range(10):
        if i == 0:
            deep = Dense(272, activation='relu')(Flatten()(input))
        else:
            deep = Dense(272, activation='relu')(deep)

    # cross layer
    cross = CrossLayer(output_dim = input.shape[2].value, num_layer = 8, name = "cross_layer")(input)

    #concat both layers
    output = merge([deep, cross], mode = 'concat')
    output = Dense(1, activation = 'sigmoid')(output)
    model = Model(inp_layer, output) 
    #print(model.summary())
    #plot_model(model, to_file = 'model.png', show_shapes = True)
    """
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
    
    if len(params) == 2:
        X_val = params[0]
        y_val = params[1]    
        model.fit([X[c] for c in train_feature],y,batch_size = 512,epochs = 1,validation_data = ([X_val[c] for c in train_feature],y_val))
    else:
        model.fit([X[c] for c in train_feature],y,batch_size = 512,epochs = 1)
    return model


def evaluate(X, y, model):
    y_pred = model.predict([X[c] for c in X.columns])
    acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y, 1)) / (y.shape[0] * 1.0)
    print("Accuracy: ", acc)


# https://keras.io/layers/writing-your-own-keras-layers/
class CrossLayer(layers.Layer):
    def __init__(self, output_dim, num_layer, **kwargs):
        self.output_dim = output_dim
        self.num_layer = num_layer
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[2]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape = [1, self.input_dim], initializer = 'glorot_uniform', name = 'w_' + str(i), trainable = True))
            self.bias.append(self.add_weight(shape = [1, self.input_dim], initializer = 'zeros', name = 'b_' + str(i), trainable = True))
        self.built = True

    def call(self, input):
        for i in range(self.num_layer):
            if i == 0:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), x), 1, keepdims = True), self.bias[i], x]))(input)
            else:
                cross = Lambda(lambda x: Add()([K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), input), 1, keepdims = True), self.bias[i], input]))(cross)
        return Flatten()(cross)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)


def process_data(data):

    #train = data[data['is_trade'].isnull() == False]
    #test =  data[data['is_trade'].isnull()]
    #types = data.dtypes
    #columns  = data.columns
    
    #C处理离散特征
    data['item_id'] = encode_feature(data['item_id'])
    
    data['user_id'] = encode_feature(data['user_id'])
    
    data['context_id'] = encode_feature(data['context_id'])
    
    data['shop_id'] = encode_feature(data['shop_id'])
    
    
    for i in range(3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else "-1"
        )
        data['category_%d'%(i)] = data['category_%d'%(i)].astype(int)
    del data['item_category_list']  
    data['category_0'] = encode_feature(data['category_0'])
    data['category_1'] = encode_feature(data['category_1'])
    data['category_2'] = encode_feature(data['category_2'])
 
    
    for i in range(3):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(
            lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else "-1"
        )
        data['predict_category_%d'%(i)] = data['predict_category_%d'%(i)].astype(int)
    del data['predict_category_property']
    data['predict_category_0'] = encode_feature(data['predict_category_0'])
    data['predict_category_1'] = encode_feature(data['predict_category_1'])
    data['predict_category_2'] = encode_feature(data['predict_category_2'])
    
    data['item_brand_id'] = encode_feature(data['item_brand_id'])
    
    data['item_city_id'] = encode_feature(data['item_brand_id'])
    
    
    data['user_occupation_id'] = encode_feature(data['user_occupation_id'])
    
    
    data['context_page_id'] = encode_feature(data['context_page_id'])
    
    
    #del data['predict_category_property'] 
    del data['item_property_list']
        #时间特征
    hour = []
    day = []
    for item in data['context_timestamp']:
        value = time.localtime(item)
        hour.append( value.tm_hour)
        day.append( value.tm_mday)
    day = np.array(day)
    hour = np.array(hour)    
    del data['context_timestamp']
    data['day'] = day
    data['hour'] = hour
    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',on=['user_id', 'day', 'hour'])



    scaler = StandardScaler()    
    cont_columns = ['item_price_level','item_sales_level','item_collected_level','item_pv_level',
                    'user_age_level','user_star_level','shop_review_num_level','shop_review_positive_rate',
                    'shop_star_level','shop_score_service','shop_score_delivery','shop_score_description','hour','user_query_day','user_query_day_hour']
    
    cate_columns = ['item_id','user_id','context_id','shop_id','item_brand_id','item_city_id',
                     'user_gender_id','user_occupation_id','context_page_id',
                     'category_0','category_1','category_2','predict_category_0','predict_category_1','predict_category_2']
    
    label = data['is_trade']
    del data['is_trade']
    data_cont = pd.DataFrame(scaler.fit_transform(data[cont_columns]), columns = cont_columns)
    data_cate = data[cate_columns]
    data = pd.concat([data.instance_id,data_cate, data_cont,data.day], axis = 1)
    
    data = pd.merge(data,train_feature)
        
    return data,label, cate_columns, cont_columns
    

if __name__ == "__main__":
    # data download from https://www.kaggle.com/uciml/forest-cover-type-dataset/data
    data = pd.read_csv("./round1_ijcai_18_train_20180301.txt",sep= " ")
    test = pd.read_csv("./round1_ijcai_18_test_a_20180301.txt",sep = " ")
    data = pd.concat([data,test], keys=['train', 'test'])
    gc.collect()
    X, y, inp_layer, inp_embed, cate_columns, cont_columns= feature_generate(data)
    del data
    gc.collect()
    
    train_feature = cate_columns
    train_feature.extend(cont_columns)
    online = False
    if online == False:
        X = pd.concat([X, y], axis = 1)
        X_train = X.loc[X.day < 24]  
        X_test = X.loc[X.day == 24]  

        y_train = X_train.is_trade
        
        X_pos = X_train.loc[X_train['is_trade'] == 1]
        X_neg = X_train.loc[X_train['is_trade'] == 0]
        X_pos = X_pos.sample(frac = 1)
        X_neg = X_neg.sample(frac = 1)
        
        y_test =  X_test.is_trade

    else:
        X = pd.concat([X, y], axis = 1)

        X_train = X.loc[X.is_trade.notnull()]
        y_train = X_train.is_trade
        X_pos = X_train.loc[X_train['is_trade'] == 1]
        X_neg = X_train.loc[X_train['is_trade'] == 0]
        X_pos = X_pos.sample(frac = 1)
        X_neg = X_neg.sample(frac = 1)    
        X_test = X.loc[X.is_trade.isnull()]
        y_test =  X_test.is_trade
    if online == False:
        model = create_model(inp_layer, inp_embed)
        for i in range(1):
            for start, end in zip(range(0, len(X_neg), 400000), range(400000, len(X_neg)+1, 400000)):
                X_train_t = pd.concat([X_pos,X_neg.iloc[start:end]])
                X_train_t = X_train_t.sample(frac = 1)
                y_train_t = X_train_t['is_trade']
                del X_train_t['is_trade']
                model = fit(model,inp_layer, inp_embed, X_train_t, y_train_t,train_feature,X_test,y_test,)
        """
        val_pre = model.predict([X_train[c] for c in train_feature],batch_size=512)[:,0]
        print("train log_loss",log_loss(y_train.values,val_pre))

        val_pre = model.predict([X_test[c] for c in train_feature],batch_size=512)[:,0]
        print("test log_loss",log_loss(y_test.values,val_pre))
        """
        
        val_pre = model.predict([X[c] for c in train_feature],batch_size=512)[:,0]
        X['predicted_score'] = val_pre
        X[['instance_id', 'predicted_score']].to_csv('deep_cross_train_baseline.csv', index=False,sep=' ')
    else:
        model = create_model(inp_layer, inp_embed)
        for i in range(1):
            for start, end in zip(range(0, len(X_neg), 400000), range(400000, len(X_neg)+1, 400000)):
                X_train_t = pd.concat([X_pos,X_neg.iloc[start:end]])
                X_train_t = X_train_t.sample(frac = 1)
                y_train_t = X_train_t['is_trade']
                del X_train_t['is_trade']
                model = fit(model,inp_layer, inp_embed, X_train_t, y_train_t,train_feature)
       
        val_pre = model.predict([X_train[c] for c in train_feature],batch_size=512)[:,0]
        X_train['predicted_score'] = val_pre
        print("train log_loss",log_loss(y_train.values,val_pre))
        
        
        val_pre = model.predict([X_test[c] for c in train_feature],batch_size=512)[:,0]
        
        test['predicted_score'] = val_pre
        
        X_train[['instance_id','is_trade', 'predicted_score']].to_csv('deep_cross_train_baseline.csv', index=False,sep=' ')
        test[['instance_id', 'predicted_score']].to_csv('deep_cross_test_baseline.csv', index=False,sep=' ')
