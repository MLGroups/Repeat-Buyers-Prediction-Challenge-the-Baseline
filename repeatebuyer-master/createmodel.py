from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVR,LinearSVR
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from getPath import *
pardir = getparentdir()

train_split_path = pardir+'/middledata/train_split1.csv'
test_split_path = pardir+'/middledata/test_split1.csv'
test_allinfo_path = pardir+'/middledata/test_all.csv'

respath = pardir+'/middledata/res.csv'

exceptcolumns = ['user_id','merchant_id','item_id','brand_id','cat_id','label']
test_except_columns = ['user_id','merchant_id','item_id','brand_id','cat_id','prob']

def getTrainData():
    data = pd.read_csv(train_split_path,encoding='utf-8')
    columns = list(data.columns.values)
    features = list(set(columns)-set(exceptcolumns))
    x = data[features]
    y = data['label']
    return x,y
    
def getPredictData():
    data = pd.read_csv(test_allinfo_path,encoding='utf-8')
    columns = list(data.columns.values)
    features = list(set(columns)-set(test_except_columns))
    x = data[features]
    ids = data[['user_id','merchant_id']]
    return x,ids

def createmodel():
    data = pd.read_csv(train_split_path,encoding='utf-8')
    columns = list(data.columns.values)
    features = list(set(columns)-set(exceptcolumns))
    x = data[features]
    y = data['label']
    clf = RandomForestClassifier(random_state=0)
    clf.fit(x, y)
    joblib.dump(clf, pardir+'/model/rf.pkl')
    del data
    clf = joblib.load(pardir+'/model/rf.pkl')
    test_data = pd.read_csv(test_split_path,encoding='utf-8')
    predict_x = test_data[features]
    predict_y = np.array(clf.predict_proba(predict_x))
    print(clf.classes_)
    y = test_data['label']
    print(roc_auc_score(y, predict_y[:,1]))
    
def predict():
    clf = joblib.load(pardir+'/model/rf.pkl')
    data = pd.read_csv(test_allinfo_path,encoding='utf-8')
    columns = list(data.columns.values)
    features = list(set(columns)-set(test_except_columns))
    x = data[features]
    y = clf.predict_proba(x)
    data['prob'] = y[:,1]
    res = pd.DataFrame({'prob':data.groupby(['user_id','merchant_id'])['prob'].max()}).reset_index()
    res.to_csv(respath,encoding='utf-8',mode = 'w', index = False)
    
if __name__=="__main__":
    # get_train_data()
    predict()

    
    
    
    
    