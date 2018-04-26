import pandas as pd
import numpy as np
from getPath import *
import gc
pardir = getparentdir()

user_log_path = pardir+'/data/user_log_format1.csv'
train_path = pardir+'/data/train_format1.csv'
test_path = pardir+'/data/test_format1.csv'

train_log_path = pardir+'/data/train_log_format1.csv'
test_log_path = pardir+'/data/test_log_format1.csv'

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

def merchantFeature(data):
    merchant = pd.DataFrame()
    merchant['item_set']=data.groupby("merchant_id")['item_id'].apply(set)
    merchant['item_num']=(merchant['item_set'].map(len)).astype(np.int16)
    merchant.drop('item_set',1,inplace=True)
    merchant['cate_set']=data.groupby("merchant_id")['cat_id'].apply(set)
    merchant['cate_num']=(merchant['cate_set'].map(len)).astype(np.int16)
    merchant.drop('cate_set',1,inplace=True)
    merchant['brand_set']=data.groupby("merchant_id")['brand_id'].apply(set)
    merchant['brand_num']=(merchant['brand_set'].map(len)).astype(np.int16)
    merchant.drop('brand_set',1,inplace=True)
    merchant['user_set']=data.groupby("merchant_id")['user_id'].apply(set)
    merchant['user_num']=(merchant['user_set'].map(len)).astype(np.int32)
    merchant.drop('user_set',1,inplace=True)

    # merchant['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    # merchant['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    # merchant['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    # merchant['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    # del group
    temp = pd.DataFrame((data.groupby(["merchant_id","user_id"])['time_stamp'].apply(set).map(len)).astype(np.int16))
    temp.reset_index(level=["merchant_id","user_id"],inplace = True)
    t = temp[temp['time_stamp']>1]
    del temp
    merchant['repeat_users']=t.groupby('merchant_id')['user_id'].count().astype(np.int16)
    del t
    merchant.reset_index(level=['merchant_id'],inplace = True)
    c = pd.DataFrame({'count':data.groupby(["merchant_id",'action_type']).size()}).reset_index()
    table = pd.pivot_table(c, values='count', index=["merchant_id"],columns=['action_type'],fill_value=0)
    table.reset_index(level=["merchant_id"],inplace = True)
    res = pd.merge(merchant,table,on="merchant_id")
    del merchant
    res.to_csv(merchant_path,encoding='utf-8',mode = 'w', index = False)
    del res
    
def oneitemfeature(data,col,path):
    c = pd.DataFrame({'count':data.groupby([col,'action_type']).size()}).reset_index()
    table = pd.pivot_table(c, values='count', index=[col],columns=['action_type'],fill_value=0)
    table.reset_index(level=[col],inplace = True)
    table.to_csv(path, encoding='utf-8',mode = 'w', index = False)
  
def itemFeature(data):
    oneitemfeature(data, "item_id", item_path)
    # item = pd.DataFrame()
    # group = data.groupby("item_id")
    # item['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    # gc.collect()
    # item['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    # gc.collect()
    # item['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    # gc.collect()
    # item['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    # del group
    # item.reset_index(level=['item_id'],inplace = True)
    # item.to_csv(item_path,encoding='utf-8',mode = 'w', index = False)
    # del item

def brandFeature(data):
    oneitemfeature(data, "brand_id", brand_path)
    # brand = pd.DataFrame()
    # group = data.groupby("brand_id")
    # brand['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    # gc.collect()
    # brand['add_to_carts'] =group.apply(lambda g:len(g[g['action_type']==1]))
    # gc.collect()
    # brand['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    # gc.collect()
    # brand['add_to_favourite'] =group.apply(lambda g:len(g[g['action_type']==3]))
    # del group
    # brand.reset_index(level=['brand_id'],inplace = True)
    # brand.to_csv(brand_path,encoding='utf-8',mode = 'w', index = False)
    # del brand 

def user_merchant_feature(data):
    user_merchant=pd.DataFrame()
    group = data.groupby(['user_id','merchant_id'])
    user_merchant['total_items']=group['item_id'].count()
    user_merchant['differnt_items'] = (group['item_id'].apply(set).map(len)).astype(np.int16)
    user_merchant['differnt_brands'] = (group['brand_id'].apply(set).map(len)).astype(np.int16)
    user_merchant.reset_index(level=['user_id','merchant_id'],inplace = True)
    user_merchant.to_csv(user_merchant_path,encoding='utf-8',mode = 'w', index = False)
    del user_merchant
    
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector):
        result = np.zeros(len(vector))
        for k in range(self.n_keys):
            result[self.indices[k]] = function(vector[self.indices[k]])
        return result
    
def one_other_feature(data, one, other, one_other_path):
    c = pd.DataFrame({'count':data.groupby([one,other,'action_type']).size()}).reset_index()
    table = pd.pivot_table(c, values='count', index=[one, other],columns=['action_type'],fill_value=0)
    table.reset_index(level=[one, other],inplace = True)
    table.to_csv(one_other_path, encoding='utf-8',mode = 'w', index = False)
    # user_item['click']=group.apply(lambda g:len(g[g['action_type']==0]))
    # gc.collect()
    # user_item['add_to_carts']=group.apply(lambda g:len(g[g['action_type']==1]))
    # gc.collect()
    # user_item['purchase']=group.apply(lambda g:len(g[g['action_type']==2]))
    # gc.collect()
    # user_item['add_to_favourite']=group.apply(lambda g:len(g[g['action_type']==3]))
    # user_item.reset_index(level=[one,other],inplace = True)
    # user_item.to_csv(one_other_path, encoding='utf-8',mode = 'w', index = False)
    # del user_item

def identify_duplicate():
    train= pd.read_csv(train_path,encoding='utf-8')
    test = pd.read_csv(test_path,encoding='utf-8')
    train_ = train.groupby(['user_id','merchant_id']).count()
    train_.reset_index(level=['user_id','merchant_id'],inplace = True)
    test_ = test.groupby(['user_id','merchant_id']).count()
    test_.reset_index(level=['user_id','merchant_id'],inplace = True)
    s1 = pd.merge(train_, test_, how='inner', on=['user_id','merchant_id'])
    print(s1)
    
def split_train_test(data):
    # train= pd.read_csv(train_path,encoding='utf-8')
    # s1 = pd.merge(train[['user_id','merchant_id']],data,how='inner', on=['user_id','merchant_id'])
    # s1.to_csv(train_log_path,encoding='utf-8',mode = 'w', index = False)
    # del s1
    test= pd.read_csv(test_path,encoding='utf-8')
    test['key']=test['user_id'].apply(str)+'_'+test['merchant_id'].apply(str)
    data['key']=data['user_id'].apply(str)+'_'+data['merchant_id'].apply(str)
    train = data[~data.key.isin(test.key)]
    train.drop('key',1, inplace = True)
    train.to_csv(train_log_path,encoding='utf-8',mode = 'w', index = False)
    del train
    s2 = pd.merge(test[['user_id','merchant_id']],data,how='inner', on=['user_id','merchant_id'])
    del test
    s2.to_csv(test_log_path,encoding='utf-8',mode = 'w', index = False)
    del s2


if __name__=="__main__":
    data = pd.read_csv(user_log_path,encoding='utf-8')
    # merchantFeature(data)
    # user_merchant_feature(data)
    # ones = ['user_id','merchant_id']
    # others = ['item_id','brand_id','cat_id']
    # path = [user_item_path,user_brand_path,user_cate_path,merchant_item_path, merchant_brand_path, merchant_cate_path]
    # i = 0
    # for one in ones:
        # for other in others:
            # one_other_feature(data, one, other, path[i])
            # i+=1
    merchantFeature(data)
    # itemFeature(data)
    # brandFeature(data)
    # split_train_test(data)
    # test()
    # train_data = pd.read_csv(train_path,encoding='utf-8')
    # newdata = pd.merge(train_data,data,how='inner', on=['user_id','merchant_id'])
    # print(newdata.groupby(['user_id','merchant_id','item_id','time_stamp','actinon_type'])
    
    


