# -*- coding: utf-8 -*-
# @Time     :2018/4/18 下午6:42
# @Author   :李二狗
# @Site     :
# @File     :first_exploration.py
# @Software :PyCharm

import os
import sys
import logging
import importlib
import argparse
import datetime
import pymongo

# importlib.reload(sys)

current_time = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    level=logging.DEBUG,
    datefmt='%a, %d %b %Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("running %s", " ".join(sys.argv))

import pandas as pd
import pymongo

train_file = pd.read_csv("data/train_format1.csv")
test_file = pd.read_csv("data/test_format1.csv")

client = pymongo.MongoClient(
    r"mongodb://admin:MzQyZDZjZWQ1Zjg@10.183.99.111:9428/admin")
db = client[r"common1"]


def basic_data_statistics():
    collection = db["base_statistic"]
    train_users_length = len(train_file['user_id'].value_counts())
    collection.insert({"train_users_length": train_users_length})
    test_users_length = len(test_file['user_id'].value_counts())
    collection.insert({"test_users_length": test_users_length})
    train_merchants_length = len(train_file['merchant_id'].value_counts())
    collection.insert({"train_merchants_length": train_merchants_length})
    test_merchants_length = len(test_file['merchant_id'].value_counts())
    collection.insert({"test_merchants_length": test_merchants_length})
    train_pairs_length = len(
        train_file.drop_duplicates(['merchant_id', 'user_id']))
    collection.insert({"train_pairs_length": train_pairs_length})
    test_pairs_length = len(
        test_file.drop_duplicates(['merchant_id', 'user_id']))
    collection.insert({"test_pairs_length": test_pairs_length})
    train_positive_pairs = len(train_file[train_file['label'] == 1])
    collection.insert({"train_positive_pairs": train_positive_pairs})
    train_positive_percent = train_positive_pairs * 1.0 / train_pairs_length
    collection.insert({"train_positive_percent": train_positive_percent})
    # 下列数据用spark算出
    log_activity_rows = 54925330
    collection.insert({"log_activity_rows": log_activity_rows})
    log_activity_users_number = 424170
    collection.insert({"log_activity_users_number": log_activity_users_number})
    log_activity_merchants_number = 4995
    collection.insert(
        {"log_activity_merchants_number": log_activity_merchants_number})
    log_activity_item_number = 1090390
    collection.insert({"log_activity_item_number": log_activity_item_number})
    log_activity_category_number = 1658
    collection.insert(
        {"log_activity_category_number": log_activity_category_number})
    log_activity_brand_number = 8443
    collection.insert({"log_activity_brand_number": log_activity_brand_number})
    # 用hive计算出的结果
    # select action_type, count(*) from tmp_whq group by action_type;
    log_activity_click_num = 48550713
    collection.insert({"log_activity_click_num": log_activity_click_num})
    log_activity_addtocart_num = 76750
    collection.insert(
        {"log_activity_addtocart_num": log_activity_addtocart_num})
    log_activity_purchase_num = 3292144
    collection.insert({"log_activity_purchase_num": log_activity_purchase_num})
    log_activity_addtofavorite_num = 3005723
    collection.insert(
        {"log_activity_addtofavorite_num": log_activity_addtofavorite_num})

    log_click_ratio = 1.0 * log_activity_click_num / log_activity_rows
    collection.insert({"log_click_ratio": log_click_ratio})
    log_addtocart_ratio = 1.0 * log_activity_addtocart_num / log_activity_rows
    collection.insert({"log_addtocart_ratio": log_addtocart_ratio})
    log_purchase_ratio = 1.0 * log_activity_purchase_num / log_activity_rows
    collection.insert({"log_purchase_ratio": log_purchase_ratio})
    log_addtofavorite_ratio = 1.0 * log_activity_addtofavorite_num / log_activity_rows
    collection.insert({"log_addtofavorite_ratio": log_addtofavorite_ratio})


if __name__ == "__main__":
    basic_data_statistics()