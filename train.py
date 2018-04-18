# -*- coding: utf-8 -*-
# @Time     :2018/4/18 下午6:47
# @Author   :李二狗
# @Site     :
# @File     :train.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator

import xgboost as xgb
from sklearn.model_selection import train_test_split


def create_feature_map(columns):
    """
    :param columns:
    :return:
    """
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in columns:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


x = pd.read_csv("x.csv")

# fmap
print("create feature map... ")
features = [f for f in x.columns if f not in [
    "label", "user_id", "seller_id", "index"]]
create_feature_map(features)


print ("start trainning ...")
print ("all samples: ", x.shape)

x0 = x[x.label == 0]
x1 = x[x.label == 1]
x0 = x0.reindex(np.random.permutation(x0.index))
x1 = x1.reindex(np.random.permutation(x1.index))

# train model
# down sample
x_posi = x1[x1.label == 1]  # [0:14952]
x_nega = x0[x0.label == 0][0:int(1.5 * 30000)]
x = pd.concat((x_posi, x_nega), axis=0)
x = x.reindex(np.random.permutation(x.index))
print ("positive samples: ", x_posi.shape)
print ("negative samples: ", x_nega.shape)


id = x[["user_id", "seller_id"]]
y = x["label"]
x = x.drop(["label"], axis=1)
x = x.drop(["user_id"], axis=1)
x = x.drop(["seller_id"], axis=1)
x = x.drop(["index"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    x.as_matrix(), y.as_matrix(), test_size=0.25)

# train sample
dtrain = xgb.DMatrix(x.as_matrix(), y.as_matrix())
print ("train samples: ", x.shape)

# # cv train
xgb_params = {
    'seed': 0,
    'colsample_bytree': 1.0,    # 0.5  0.6  0.7  0.8  0.9
    'silent': 1,
    'subsample': 0.5,
    'learning_rate': 0.05,
    'objective': 'reg:logistic',
    'max_depth': 4,     # 最优  4 、5
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'loss': 'reg:logistic',
    'eval_metric': 'auc',
    'max_delta_step': 0
}
res = xgb.cv(xgb_params, dtrain, num_boost_round=400, nfold=4, stratified=False,
             early_stopping_rounds=25, verbose_eval=5, show_stdv=True)

# # train for online
xgb_params = {
    # general parameters
    'booster': 'gbtree',
    'silent': 1,

    # booster parameters
    'learning_rate': 0.05,
    'gamma': 0,
    'max_depth': 4,
    'min_child_weight': 1,
    'max_delta_step': 0,    # 每棵树权重更新的最大值
    'subsample': 0.5,
    'colsample_bytree': 1.0,

    # task parameters
    'objective': 'reg:logistic',
    'eval_metric': 'auc',
    'seed': 0,
}
gbdt = xgb.train(xgb_params,
                 dtrain=dtrain,
                 num_boost_round=400
                 )
importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', figsize=(10, 25))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
print(df.sort_values(by='fscore', ascending=False))

# tx_posi = x1[x1.label==1][14952:]
# tx_nega = x0[x0.label==0][int(1.5*30000) + 1:]
# print "test positive sample: ", tx_posi.shape
# print "test negative sample: ", tx_nega.shape
#
# x_test = pd.concat((tx_nega, tx_posi), axis=0)
# x_test = x_test.reindex(np.random.permutation(x_test.index))
# x_test_label = x_test["label"]
# x_test = x_test.drop(["label"], axis=1)
#
#
# print "test samples: ", x_test.shape
# pred = gbdt.predict(xgb.DMatrix(x_test.as_matrix()))
# pred_y = [0 if i < 0.5 else 1 for i in pred]
# from sklearn.metrics import roc_auc_score
# x_test_label = x_test_label.tolist()
# auc = roc_auc_score(x_test_label, pred)
# print (type(pred))
# print (auc)
# print(pred)

# #roc and auc
# from sklearn.metrics import auc
# from sklearn.metrics import roc_curve
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# fpr, tpr, thresholds = roc_curve(x_test_label, pred, pos_label=1)
# plt.figure()
# plt.plot([0,1], [0,1], 'k--')
# plt.plot(fpr, tpr, label='xgb')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.show()


# online data

x_online = pd.read_csv("x_online.csv")
x_online = x_online.drop(["prob"], axis=1)
id = x_online[["user_id", "seller_id"]]
x_online = x_online.drop(["user_id"], axis=1)
x_online = x_online.drop(["seller_id"], axis=1)
x_online = x_online.drop(["index"], axis=1)
print ("online sample shape:", x_online.shape)

pred_online = gbdt.predict(xgb.DMatrix(x_online.as_matrix()))
pred_online = pd.DataFrame({"prob": pred_online})
pred_online = pd.concat((id, pred_online), axis=1)
pred_online.to_csv("online.csv", index=False)