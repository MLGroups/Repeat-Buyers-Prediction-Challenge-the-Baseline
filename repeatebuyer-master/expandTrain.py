import pandas as pd
import numpy as np
from getPath import *
pardir = getparentdir()

train_path = pardir + '/data/test_format1.csv'
user_log_path = pardir+'/data/user_log_format1.csv'
expand_path = pardir+'/middledata/test_data.csv'

def expand():
    data = pd.read_csv(user_log_path,encoding='utf-8')
    total = pd.DataFrame(data.groupby(['merchant_id','item_id'])['brand_id','cat_id'].first())
    del data
    total.reset_index(level=['merchant_id','item_id'],inplace = True)
    train_data = pd.read_csv(train_path,encoding='utf-8')
    res = pd.merge(train_data, total, on=['merchant_id'])
    del train_data,total
    res.to_csv(expand_path,encoding='utf-8',mode = 'w', index = False)
    
if __name__=="__main__":
    expand()
    
  
    
    