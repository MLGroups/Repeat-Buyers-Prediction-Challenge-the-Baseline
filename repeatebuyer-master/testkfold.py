from sklearn.model_selection import KFold
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[4,5]])
y = np.array([1, 2, 3, 4,5])
kf = KFold(n_splits=4,random_state=None, shuffle=False)
kf.get_n_splits(X)
# print(kf.split(X))  
# KFold(n_splits=4, random_state=None, shuffle=False)
train=[]
test=[]
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    train.append(list(train_index))
    print(X[train_index])
    test.append(list(test_index))
print(train)
print(test)
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
