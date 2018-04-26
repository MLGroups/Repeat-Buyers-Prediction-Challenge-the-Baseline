import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from onehotencoder import *
from getPath import *
import os
pardir = getparentdir()

merchant_path = pardir +'/middledata/merchant.csv'
item_path = pardir +'/middledata/item.csv'
brand_path = pardir +'/middledata/brand.csv'
user_merchant_path = pardir+'/middledata/user_merchant.csv'
user_item_path = pardir+'/middledata/user_item.csv'
user_brand_path = pardir+'/middledata/user_brand.csv'
user_cate_path = pardir+'/middledata/user_cate.csv'
merchant_item_path = pardir+'/middledata/merchant_item.csv'
merchant_brand_path = pardir+'/middledata/merchant_brand.csv'
merchant_cate_path = pardir+'/middledata/merchant_cate.csv'
user_path = pardir+'/data/user_info_format1.csv'

train_path = pardir+'/middledata/train_data.csv'
test_path = pardir+'/middledata/test_data.csv'

test_allinfo_path = pardir+'/middledata/test_all.csv'

train_split_path = pardir+'/middledata/train_split.csv'
test_split_path = pardir+'/middledata/test_split.csv'
user_info_code_path = pardir+'/middledata/user_info_encode.csv'

def get_user_data():
    if os.path.exists(user_info_code_path):
        userinfo = pd.read_csv(user_info_code_path,encoding='utf-8')
        return userinfo
        
    data = pd.read_csv(user_path, encoding='utf-8')
    agerange = 9
    ageonecode = encodebins(agerange)
    data['age_range'].fillna(0,inplace=True)
    data['gender'].fillna(2,inplace=True)

    sexrange = 3
    sexcode = encodebins(sexrange)
    for i in range(sexrange):
        data["sex"+str(i)] = data['gender'].map(lambda x:sexcode[int(x)][i])
        columns.append("sex"+str(i))
    data[columns].to_csv(user_info_code_path, encoding='utf-8',mode = 'w',index = False)
    return data[columns]

def combineFeatures(isTest):
    path = train_path
    if isTest:
        path = test_path
    train_data = pd.read_csv(path,encoding='utf-8')
    merchant_data = pd.read_csv(merchant_path,encoding='utf-8')
    first = pd.merge(train_data, merchant_data, on="merchant_id")
    del train_data,merchant_data
    item_data =pd.read_csv(item_path,encoding='utf-8')
    second = pd.merge(first, item_data, on="item_id")
    del item_data,first
    brand_data = pd.read_csv(brand_path,encoding='utf-8')
    third = pd.merge(second, brand_data, on = "brand_id")
    del brand_data,second
    user_merchant_data =pd.read_csv(user_merchant_path,encoding='utf-8')
    fourth = pd.merge(third, user_merchant_data, on=['user_id','merchant_id'])
    del user_merchant_data,third
    user_item_data =pd.read_csv(user_item_path,encoding='utf-8')
    fifth = pd.merge(fourth, user_item_data, on=['user_id','item_id'])
    del user_item_data,fourth
    user_brand_data =pd.read_csv(user_brand_path,encoding='utf-8')
    sixth = pd.merge(fifth, user_brand_data, on=['user_id','brand_id'])
    del user_brand_data,fifth
    user_cate_data =pd.read_csv(user_cate_path,encoding='utf-8')
    seventh = pd.merge(sixth, user_cate_data, on=['user_id','cat_id'])
    del user_cate_data,sixth
    merchant_item_data =pd.read_csv(merchant_item_path,encoding='utf-8')
    eighth = pd.merge(seventh, merchant_item_data, on=['merchant_id','item_id'])
    del merchant_item_data,seventh
    merchant_brand_data =pd.read_csv(merchant_brand_path,encoding='utf-8')
    ninth = pd.merge(eighth, merchant_brand_data, on=['merchant_id','brand_id'])
    del merchant_brand_data,eighth
    merchant_cate_data = pd.read_csv(merchant_cate_path,encoding='utf-8')
    tenth = pd.merge(ninth, merchant_cate_data, on=['merchant_id','cat_id'])
    del ninth,merchant_cate_data
    user_data = get_user_data()
    elenvth = pd.merge(user_data, tenth, on=['user_id'])
    if isTest:
        elenvth.to_csv(test_allinfo_path, encoding='utf-8',mode = 'w', index = False )
    else:
        X_train,X_test = sampleTest(elenvth)
        X_train.to_csv(train_split_path,encoding='utf-8',mode = 'w', index = False)
        X_test.to_csv(test_split_path,encoding='utf-8',mode = 'w', index = False)

def sampleTest(list):
    X_train, X_test= train_test_split(list, test_size=0.1,random_state=2)
    return X_train,X_test
     
if __name__=="__main__":
    combineFeatures(0)
    combineFeatures(1)
    # train = pd.read_csv(merchant_path,encoding='utf-8')
    # print(train[train['merchant_id']==4044])
    
    
    
    
    
    
    