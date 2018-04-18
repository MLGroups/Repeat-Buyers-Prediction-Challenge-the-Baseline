# -*- coding: utf-8 -*-
# @Time     :2018/4/18 下午6:46
# @Author   :李二狗
# @Site     :
# @File     :analysis3.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("data/test_format1.csv")
user_log = pd.read_csv("data/user_log_format1.csv")
user_log = user_log.fillna(0)

# 用户-商家整体交易特征提取
repeat_buy_user = pd.read_csv(
    "data/repeat_buy_user.csv")
a = pd.DataFrame({"user_id": repeat_buy_user.user_id, "merchant_id": repeat_buy_user.seller_id,
                  "repeat_buy_times_user_seller": repeat_buy_user.action_type})
b = pd.merge(train, a, on=["user_id", "merchant_id"], how='left')

x = b
# rename merchant_id to seller_id
x = x.rename(columns={"merchant_id": "seller_id"})
x = x.rename(
    columns={"repeat_buy_times_user_seller": "buy_times_between_user_and_seller"})
# 新加特征 2017-04-21
# 用户在该商家购买是否大于1次、大于2次、、、分别提取特征
x["user_buy_once_in_this_seller"] = 0
x["user_buy_twice_in_this_seller"] = 0
x["user_buy_thrice_in_this_seller"] = 0
x["user_buy_fourtime_in_this_seller"] = 0
x["user_buy_fivetime_in_this_seller"] = 0
x["user_buy_sixtime_in_this_seller"] = 0
x["user_buy_more_than_sixtime_in_this_seller"] = 0
x.loc[x.buy_times_between_user_and_seller ==
      1, "user_buy_once_in_this_seller"] = 1
x.loc[x.buy_times_between_user_and_seller ==
      2, "user_buy_twice_in_this_seller"] = 1
x.loc[x.buy_times_between_user_and_seller ==
      3, "user_buy_thrice_in_this_seller"] = 1
x.loc[x.buy_times_between_user_and_seller ==
      4, "user_buy_fourtime_in_this_seller"] = 1
x.loc[x.buy_times_between_user_and_seller ==
      4, "user_buy_fivetime_in_this_seller"] = 1
x.loc[x.buy_times_between_user_and_seller ==
      5, "user_buy_sixtime_in_this_seller"] = 1
x.loc[x.buy_times_between_user_and_seller > 6,
      "user_buy_more_than_sixtime_in_this_seller"] = 1


# 新增加特征 2017-04-21
# 用户对商家的favourite
a = user_log[user_log.action_type == 3].groupby(
    ["user_id", "seller_id"]).count().reset_index()
a = pd.DataFrame({"user_id": a.user_id, "seller_id": a.seller_id,
                  "favourite_times_bettwen_user_and_seller": a.item_id})
x = pd.merge(x, a, on=['user_id', 'seller_id'], how='left')   # NaN表示没有

# 添加用户整体的重复购买、商家整体的重复购买
# 商家的整体重复购买
a = user_log[user_log.action_type == 2].groupby(
    ['seller_id', 'user_id']).count()
a = a.reset_index()
# 考察商家被多少个用户多次购买过(购买次数>2)
seller_be_repeat_buy_by_how_many_user = a[a.action_type > 1].groupby([
                                                                     "seller_id"]).count()
seller_be_repeat_buy_by_how_many_user = seller_be_repeat_buy_by_how_many_user.reset_index()
seller_be_repeat_buy_by_how_many_user["seller_be_repeat_buy_by_how_many_user"] = seller_be_repeat_buy_by_how_many_user.item_id
seller_be_repeat_buy_by_how_many_user = seller_be_repeat_buy_by_how_many_user.drop([
                                                                                   "item_id"], axis=1)
seller_be_repeat_buy_by_how_many_user = seller_be_repeat_buy_by_how_many_user.drop(
    ["user_id", "cat_id", "time_stamp", "brand_id", "action_type"], axis=1)
# 考察商家被重复购买的用户重复购买过多少次
seller_be_repeat_buy_all_times_by_repeat_user = a[a.action_type > 1].groupby(
    ["seller_id"]).agg({"item_id": pd.np.sum})
seller_be_repeat_buy_all_times_by_repeat_user = seller_be_repeat_buy_all_times_by_repeat_user.reset_index()
seller_be_repeat_buy_all_times_by_repeat_user[
    "seller_be_repeat_buy_all_times_by_repeat_user"] = seller_be_repeat_buy_all_times_by_repeat_user.item_id
seller_be_repeat_buy_all_times_by_repeat_user = seller_be_repeat_buy_all_times_by_repeat_user.drop([
                                                                                                   "item_id"], axis=1)
# drop columns
# merge data
x = pd.merge(x, seller_be_repeat_buy_by_how_many_user,
             on=["seller_id"], how='left')
x = pd.merge(x, seller_be_repeat_buy_all_times_by_repeat_user,
             on=["seller_id"], how='left')
x["seller_be_repeat_buy_times_on_avg_user"] = x.seller_be_repeat_buy_all_times_by_repeat_user * \
    1.0 / x.seller_be_repeat_buy_by_how_many_user


# 用户的整体重复购买
a = user_log[user_log.action_type == 2].groupby(
    ['user_id', 'seller_id']).count()
a = a.reset_index()
b = a[a.action_type > 1]  # 过滤点没有重复购买记录的user_id - seller_id 对

b_user_repeat_buy_sellers = b[["user_id", "seller_id"]].groupby([
                                                                "user_id"]).count()
b_user_repeat_buy_sellers = b_user_repeat_buy_sellers.reset_index()
b_user_repeat_buy_sellers['how_many_sellers_a_user_repeat_buy'] = b_user_repeat_buy_sellers.seller_id
b_user_repeat_buy_sellers = b_user_repeat_buy_sellers.drop(
    ["seller_id"], axis=1)

b_user_repeat_buy_times = b[["user_id", "item_id"]].groupby(
    ["user_id"]).agg({"item_id": pd.np.sum})  # 不会按照seller去重的
b_user_repeat_buy_times = b_user_repeat_buy_times.reset_index()
b_user_repeat_buy_times['how_many_times_a_user_repeat_buy'] = b_user_repeat_buy_times.item_id
b_user_repeat_buy_times = b_user_repeat_buy_times.drop(["item_id"], axis=1)

x = pd.merge(x, b_user_repeat_buy_sellers, on=['user_id'], how='left')
x = pd.merge(x, b_user_repeat_buy_times, on=['user_id'], how='left')
x['how_many_avg_times_a_user_buy_for_each_seller'] = x.how_many_times_a_user_repeat_buy * \
    1.0 / x.how_many_sellers_a_user_repeat_buy
x['dose_a_user_has_repeat_buy_habit'] = 0
x.dose_a_user_has_repeat_buy_habit[(
    x.how_many_times_a_user_repeat_buy.notnull())] = 1

# 构造一个新的特征
x["buy_times_between_user_and_seller_div_how_many_times_a_user_repeat_buy"] = x.buy_times_between_user_and_seller * \
    1.0 / x.how_many_times_a_user_repeat_buy


# 用户-商家在11.11以前的购买交易行为
a = user_log[user_log.time_stamp != 1111]
a = a[a.action_type == 2]
b = a.groupby(["user_id", "seller_id"]).count()
b = b.reset_index()
buy_times_between_user_and_seller_befor_1111 = pd.DataFrame(
    {"user_id": b.user_id, "seller_id": b.seller_id, "buy_times_between_user_and_seller_befor_1111": b.item_id})
x = pd.merge(x, buy_times_between_user_and_seller_befor_1111,
             on=["user_id", "seller_id"], how='left')

# 用户-商家在11.11以前的favourite行为
a = user_log[user_log.time_stamp != 1111]
a = a[a.action_type == 3]
b = a.groupby(["user_id", "seller_id"]).count()
b = b.reset_index()
favourite_times_between_user_and_seller_befor_1111 = pd.DataFrame(
    {"user_id": b.user_id, "seller_id": b.seller_id, "favourite_times_between_user_and_seller_befor_1111": b.item_id})
x = pd.merge(x, favourite_times_between_user_and_seller_befor_1111,
             on=["user_id", "seller_id"], how='left')


# 用户-品牌-商家特征提取
# 这里最好自己定义一个函数，计算用户喜欢品牌的商家出售品牌的之间的匹配度，但是计算量有点大
# 计算 用户-商家 匹配度 by brand
# 用户从这个商家购买某个品牌下的商品数目
a = user_log[user_log.action_type == 2].groupby(
    ["user_id", "seller_id", "brand_id"]).count().reset_index()
"""
In [57]: a.head()
Out[57]:
   user_id  seller_id  brand_id  item_id  cat_id  time_stamp  action_type
0        1        925    7402.0        1       1           1            1
1        1       1019    6805.0        4       4           4            4   # 用户1在商家1019购买品牌6805.0有4次
2        1       4026    1469.0        1       1           1            1
3        2        420    4953.0        3       3           3            3
4        2       1179    8120.0        1       1           1            1
"""
a = a[["user_id", "seller_id", "brand_id", "item_id"]]
aa = user_log[user_log.action_type == 2].groupby(
    ["seller_id", "brand_id"]).count().reset_index()
"""
In [61]: aa.head()
Out[61]:
   seller_id  brand_id  user_id  item_id  cat_id  time_stamp  action_type
0          1    1104.0      326      326     326         326          326
1          1    1662.0    17379    17379   17379       17379        17379     # 商家1的品牌1662.0被购买过17379次
2          2    2921.0      189      189     189         189          189
3          3     970.0       67       67      67          67           67
4          4       0.0        2        2       2           2            2
"""
aa = pd.DataFrame({"seller_id": aa.seller_id, "brand_id": aa.brand_id,
                   "number_of_brand_this_seller_sells": aa.item_id})  # 商家销售某个品牌下商品的数目
a = pd.merge(a, aa, on=["seller_id", "brand_id"], how='left')
a["feat1"] = a.item_id * a.number_of_brand_this_seller_sells   # 构造一个新的特征
# a["feat2"] = x.buy_times_between_user_and_seller * a.number_of_brand_this_seller_sells
# 在用户从商家购买的所有品牌销售量之和
b = a.groupby(["user_id", "seller_id"]).agg(
    {"number_of_brand_this_seller_sells": pd.np.sum}).reset_index()
sell_number_of_all_brand_this_user_buy_from_this_seller = pd.DataFrame(
    {"user_id": b.user_id, "seller_id": b.seller_id, "sell_number_of_all_brand_this_user_buy_from_this_seller": b.number_of_brand_this_seller_sells})
x = pd.merge(x, sell_number_of_all_brand_this_user_buy_from_this_seller, on=[
             'user_id', 'seller_id'], how='left')
# 在对feat1做一次加工后, 并入到x中
b = a.groupby(["user_id", "seller_id"]).agg({"feat1": pd.np.sum}).reset_index()
x = pd.merge(x, b, on=["user_id", "seller_id"], how='left')
# 计算user 从 seller 购买过几种类型的商品
b = a.groupby(["user_id", "seller_id"]).count().reset_index()
how_many_kind_of_brand_user_buy_from_seller = pd.DataFrame(
    {"user_id": b.user_id, "seller_id": b.seller_id, "how_many_kind_of_brand_user_buy_from_seller": b.item_id})
x = pd.merge(x, how_many_kind_of_brand_user_buy_from_seller,
             on=["user_id", "seller_id"], how='left')
x = x.fillna(0)

# 用户在某个商家购买过的品牌数目站用户总购买过品牌的数量的比例
a = user_log[user_log.action_type == 2].groupby(
    ["user_id", "brand_id"]).count().reset_index()
a = a.groupby(["user_id"]).count().reset_index()
# 用户购买的总的diff品牌数目
b = pd.DataFrame(
    {"user_id": a.user_id, "number_of_diff_brand_this_user_buy": a.brand_id})
x = pd.merge(x, b, on=["user_id"], how='left')
x["user_buy_diff_brand_from_this_seller_div_user_buy_all_diff_brand"] = x.how_many_kind_of_brand_user_buy_from_seller * \
    1.0 / x.number_of_diff_brand_this_user_buy


# 新增加特征 2017-04-21
# 添加用户基本属性
user_info = pd.read_csv(
    "data/user_info_format1.csv").reset_index()
x = pd.merge(x, user_info, on=['user_id'], how='left')
# 添加用户的总购物次数
a = user_log[user_log.action_type == 2].groupby(
    ["user_id"]).count().reset_index()
user_whole_buy_times = pd.DataFrame(
    {"user_id": a.user_id, "user_whole_buy_times": a.item_id})
x = pd.merge(x, user_whole_buy_times, on=['user_id'], how='left')
# 用户在某个商家购买的物品数目站用户总的购买物品的比例
x["user_buy_times_from_this_seller_ratio_to_user_whole_buy_times"] = x.buy_times_between_user_and_seller * \
    1.0 / x.user_whole_buy_times


# 用户-类目-商家 特征提取
# 这些特征最终都要经过group by 聚合成用户-商家特征
a_ucs = user_log[user_log.action_type == 2].groupby(
    ["user_id", "cat_id", "seller_id"]).count().reset_index()
a_ucs = pd.DataFrame({"user_id": a_ucs.user_id, "cat_id": a_ucs.cat_id,
                      "seller_id": a_ucs.seller_id, "ucs": a_ucs.item_id})
a_uc = user_log[user_log.action_type == 2].groupby(
    ["user_id", "cat_id"]).count().reset_index()
a_uc = pd.DataFrame(
    {"user_id": a_uc.user_id, "cat_id": a_uc.cat_id, "uc": a_uc.item_id})
a_cs = user_log[user_log.action_type == 2].groupby(
    ["seller_id", "cat_id"]).count().reset_index()
a_cs = pd.DataFrame({"seller_id": a_cs.seller_id,
                     "cat_id": a_cs.cat_id, "cs": a_cs.item_id})
a = pd.merge(a_ucs, a_uc, on=['user_id', 'cat_id'], how='left')
a = pd.merge(a, a_cs, on=['seller_id', 'cat_id'], how='left')
"""
In [39]: a
Out[39]:
         cat_id  seller_id  ucs  user_id  uc    cs
0           992       1019    4        1   4   577
1          1023        925    1        1   1  1220
2          1252       4026    1        1   1  1068
3           420       1784    2        2   2   881     # 用户1784在商家2购买国过类目420下的商品2次
                                                       # 商家2类目420下的商品销售共计881
                                                       # 用户1784购买类目420下的商品共计2次
4           500       1179    1        2   1    49
5           602        420    2        2   2  4405
6           703       2076    1        2   1    15
7           737       1974    1        2   1   394
8          1130       1679    3        2   3   926
9          1142       4924    1        2   1   147
10         1213        420    1        2   1  1326
11         1401       3552    2        2   2  2400
12          606       2313    1        3   1   916
13         1134       4461    1        3   1    56
"""
a["ucs_div_uc"] = a.ucs * 1.0 / \
    a.uc   # 用户在商家购买类目下商品的数量占用户购买该类目下商品数量的比,越大证明用户越喜欢在该商家购买类目下的商品
a["ucs_multiply_cs"] = a.ucs * a.cs  # 新构造的特征, 业务含义待定
# merge to x
b = a.groupby(["user_id", "seller_id"]).agg(
    {"ucs_div_uc": pd.np.sum}).reset_index()
x = pd.merge(x, b, on=['user_id', 'seller_id'], how='left')
b = a.groupby(["user_id", "seller_id"]).agg(
    {"ucs_multiply_cs": pd.np.sum}).reset_index()
x = pd.merge(x, b, on=['user_id', 'seller_id'], how='left')


# 用户-商家的加入购物车特征
# add_cart_time_bettwen_user_and_seller很多都为空
a = user_log[user_log.action_type == 1].groupby(
    ["user_id", "seller_id"]).count().reset_index()
a = pd.DataFrame({"user_id": a.user_id, "seller_id": a.seller_id,
                  "add_cart_time_bettwen_user_and_seller": a.item_id})
x = pd.merge(x, a, on=["user_id", "seller_id"], how='left')


# 用户点击率行为特征
user_profile = pd.read_csv(
    "data/user_profile.csv")
user_profile = user_profile.drop(["_id"], axis=1)
x = pd.merge(x, user_profile, on=["user_id"], how='left')

seller_profile = pd.read_csv(
    "data/merchant_profile.csv")
seller_profile = seller_profile.rename(columns={"merchant_id": "seller_id"})
x = pd.merge(x, seller_profile, on=["seller_id"], how='left')

# 2017-05-05
# 用户-商家点击购买比
a = user_log[user_log.action_type == 2].groupby(
    ["user_id", "seller_id"]).count().reset_index()
b = user_log[user_log.action_type == 0].groupby(
    ["user_id", "seller_id"]).count().reset_index()
a = pd.DataFrame(
    {"user_id": a.user_id, "seller_id": a.seller_id, "a": a.item_id})
b = pd.DataFrame(
    {"user_id": b.user_id, "seller_id": b.seller_id, "b": b.item_id})
b = pd.merge(b, a, on=["user_id", "seller_id"], how='left')
b = b.fillna(0)
b["user_seller_buy_click_ratio"] = b.a * 1.0 / (b.a + b.b)
b = b.drop((["a", "b"]), axis=1)
x = pd.merge(x, b, on=["user_id", "seller_id"], how='left')
x = x.fillna(0)   # train data


x.to_csv('x_online.csv', index=False)  # online data
