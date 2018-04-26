from sklearn.model_selection import KFold
from createmodel import *
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

stacking_res_path = pardir+'/middledata/stackingres.csv'

def get_k_fold(data):
    kf = KFold(n_splits=5,random_state=1,shuffle=True)
    train_indexs = []
    test_indexs = []
    for train_index, test_index in kf.split(data):
        train_indexs.append(train_index)
        test_indexs.append(test_index)
    return train_indexs,test_indexs
    
def createmodel():
    x,y = getTrainData()
    test_data,ids = getPredictData()
    train_indexs,test_indexs = get_k_fold(x)
    length = len(x)
    firstlayer = np.array([1.0]*length)
    #stacking linearsvr
    x_arr = np.array(x)
    y_arr = np.array(y)
    for i in range(len(train_indexs)):
        print("first train"+str(i))
        train = x_arr[train_indexs[i]]
        label = y_arr[train_indexs[i]]
        path = pardir+'/model/lr'+str(i)+".pkl"
        if os.path.exists(path):
            regr = joblib.load(pardir+'/model/lr'+str(i)+".pkl")
        else:
            regr = LinearRegression(normalize=True)
            regr.fit(train,label)
            joblib.dump(regr, pardir+'/model/lr'+str(i)+".pkl")
        res = regr.predict(x_arr[test_indexs[i]])
        firstlayer[test_indexs[i]] = res
        # res = regr.predict(test_data)
        # test.append(res)
    firstlayer = np.array([[f] for f in firstlayer])
    path = pardir+'/model/lrtest.pkl'
    if os.path.exists(path):
        clf = joblib.load(path)
    else:
        clf = LinearRegression(normalize=True)
        clf.fit(x_arr,y_arr)
        joblib.dump(clf, path)
    test1 = clf.predict(test_data)
    test1 = np.array([[t] for t in test1])
    # test = np.mean(test,axis = 0)
    # test = [[t] for t in test]
    # firstlayer = x_arr
    secondlayer = np.array([1.0]*length)
    finalres = []
    for i in range(len(train_indexs)):
        print("second train"+str(i))
        train = x_arr[train_indexs[i]]
        label = y_arr[train_indexs[i]]
        path = pardir+'/model/rf'+str(i)+".pkl"
        if os.path.exists(path):
            regr = joblib.load(path)
        else:
            clf = RandomForestClassifier(random_state=0)
            clf.fit(train,label)
            joblib.dump(clf, path)
        res = clf.predict(x_arr[test_indexs[i]])
        secondlayer[test_indexs[i]] = res
    secondlayer = np.array([[f] for f in secondlayer])
    path = pardir+'/model/rftest.pkl'
    if os.path.exists(path):
        clf = joblib.load(path)
    else:
        clf = RandomForestClassifier(random_state=0)
        clf.fit(x_arr,y_arr)
        joblib.dump(clf, path)
    test2 = (clf.predict_proba(test_data))[:,1]
    test2 = np.array([[t] for t in test2])
    # res = np.mean(test,axis = 0
    
    train = np.hstack((x_arr,firstlayer,secondlayer))
    test = np.hstack((test_data,test1,test2))
    path = pardir+'/model/gbdt.pkl'
    if not os.path.exists(path):
        clf =  GradientBoostingClassifier()
        clf.fit(train, y_arr)
        joblib.dump(clf, path)
    else:
        clf = joblib.load(path)
  
    predict_res = clf.predict_proba(test)
    # print(clf.classes_)
    # print(predict_res)
    ids['prob'] = predict_res[:,1]
    res = pd.DataFrame({'prob':ids.groupby(['user_id','merchant_id'])['prob'].max()}).reset_index()
    res.to_csv(stacking_res_path,encoding='utf-8',mode = 'w', index = False)
 
if __name__=="__main__":
    createmodel()
        
        
        
    
