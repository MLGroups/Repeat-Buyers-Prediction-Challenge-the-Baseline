# -*- coding: utf-8 -*-
# @Time     :2018/4/24 下午3:16
# @Author   :李二狗
# @Site     :
# @File     :find_feature.py
# @Software :PyCharm

import pandas as pd
import numpy as np

# 分为 7种关系  user,item,seller,user-item,user-seller,item-seller,user-item-seller

##  user  relate feature ##
"""
u1 用户操作时间 user_handle_day_count
u2 用户性别
u4_x user action    user_action_0_count,user_action_1_count,user_action_2_count,user_action_3_count
u5 用户年龄段  user_age_type
u6 用户操作了几天

"""

##  item  relate feature ##
"""
i1 该 cat_id 下面有多少item  cat_item_count

"""

## seller relate feature ##
"""
s1 该 seller_id 有多少 brand seller_brand_count
"""

## user- item relate feature ##
"""
ui1  (1+4)用户在此item 上总共做了多少次操作,每个action做了多少次。 action ui_action_type_total_count 
ui_action_type_hot_0_count  ui_action_type_hot_1_count  ui_action_type_hot_2_count ui_action_type_hot_3_count
 
"""


## user- seller relate feature ##
"""
us1  (1+4)用户在此seller 上总共做了多少次操作,每个action做了多少次。 ss_action_type_total_count 
ss_action_type_hot_0_count  ss_action_type_hot_1_count  ss_action_type_hot_2_count ss_action_type_hot_3_count

us2  (1+4)用户在此seller的 brand 上总共做了多少次操作,每个action做了多少次。 ss2_action_type_total_count 
ss_action_type_hot_0_count  ss_action_type_hot_1_count  ss_action_type_hot_2_count ss_action_type_hot_3_count
 
"""

## item- seller relate feature ##
"""
si1  该 seller 有多少 item  seller_item_count
#r_si1  一一对应 忽略 

si2  该 seller 有多少 cat  seller_cat_count
r_si2 该 cat 有多少 seller

si3  该 seller的 brand 有多少 item  seller_brand_item_count
r_si3

si4  该 seller的 brand 有多少 cat  seller_brand_cat_count
r_si4


"""

#  user- seller - item 关系
"""
usi1 用户在此商户的 的某item上面操作了多少次，usi_action_type_total_count 
usi_action_type_hot_0_count  usi_action_type_hot_1_count  usi_action_type_hot_2_count usi_action_type_hot_3_count
r_usi1
usi2 用户在此商户的brand 的某item上面操作了多少次，usi_action_type_total_count 
usi_action_type_hot_0_count  usi_action_type_hot_1_count  usi_action_type_hot_2_count usi_action_type_hot_3_count
r_usi2
"""




user_log_path = 'data/user_log_format1.csv'
user_log_dataset = pd.read_csv(user_log_path, header=None)
print(user_log_dataset.head())

# u1 用户操作时间 user_handle_day_count