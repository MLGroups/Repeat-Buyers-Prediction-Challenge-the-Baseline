import pandas as pd
from getPath import *
pardir = getparentdir()

def analyze_train():
    data = pd.read_csv(pardir+'/data/user_log_format1.csv')
    countdf = pd.DataFrame({'count':data.groupby("item_id")['merchant_id'].nunique()}).reset_index()
    morethanone = countdf['item_id'][countdf['count']>1]
    print(morethanone)
    # users = len(data.groupby('user_id').size())
    # merchants = len(data.groupby('merchant_id').size())
    # positives = data['label'][data['label']==1]
    # data.rename(index=str, columns={'seller_id':'merchant_id'}, inplace=True)
    # data.to_csv(pardir+'/data/user_log_format1.csv',encoding='utf-8',mode = 'w', index = False)
    # del data
    # print(data)
    # print(users)
    # print(merchants)
    # print(len(positives))
    # print(len(data))
    # print(len(positives)/len(data))
    
def analyze_train_label():
    data = pd.read_csv(pardir+'/rawdata/data_format2/train_format2.csv')
    print(len(data))
    print(len(data[data['label']==-1]))
    
def analyze_train_data():
    data = pd.read_csv(pardir+'/middledata/train_split1.csv')
    countdf = pd.DataFrame({'count':data.groupby(["merchant_id","user_id"])['label'].nunique()}).reset_index()
    morethanone = countdf[["merchant_id","user_id"]][countdf['count']>1]
    print(morethanone)
    
analyze_train_data()
    



