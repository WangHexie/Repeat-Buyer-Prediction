# coding: utf-8

# In[1]:


from typing import List, Any, Union

import matplotlib
from pandas import DataFrame
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from deep_learning.data_convert import *
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from basic_model.feature_extraction.user_profile import operate_days, purchase_days
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from feature_selector import FeatureSelector
import lightgbm as lgb
import os
import functools
import joblib


# In[3]:


def read_feature_file(file_name) -> pd.DataFrame:
    print("Reading:", file_name)
    file_path = os.path.join(get_root_path(), "feature_vectors", file_name)
    df = pd.read_csv(file_path, index_col=0)
    print("feature length:", len(df.columns.values))
    return df


def select_features_without_label(features: pd.DataFrame, missing_threshold=0.95,
                                  correlation_threshold=1.0) -> pd.DataFrame:
    fs = FeatureSelector(data=features)
    fs.identify_missing(missing_threshold)
    fs.identify_single_unique()
    #     fs.identify_collinear(correlation_threshold)
    #     return fs.remove(methods=['missing', 'single_unique', "collinear"])
    return fs.remove(methods=['missing', 'single_unique'])


def read_all_features(file_name_list: list) -> list:
    return [select_features_without_label(read_feature_file(file_name)) for file_name in file_name_list]


def merge_all_features_on_index(feature_df_list: list) -> pd.DataFrame:
    if len(feature_df_list) == 0:
        return pd.DataFrame()
    first_one = feature_df_list.pop()
    time_suffix = 0
    for feature_df in feature_df_list:
        time_suffix += 1
        first_one = first_one.join(feature_df, how="outer", rsuffix="_" + str(time_suffix))
    return first_one


def feature_transform_pipeline(file_name_list: list, dic_list: list = None, df_list: list = None,
                               choose_to_downcast=True) -> pd.DataFrame:
    features_df_list: List[DataFrame] = read_all_features(file_name_list) + [
        select_features_without_label(pd.DataFrame.from_dict(i).T)
        for i in dic_list] if dic_list is not None else [] + [
        select_features_without_label(i) for i
        in df_list] if df_list is not None else []
    features_df = merge_all_features_on_index(features_df_list)
    if choose_to_downcast:
        return select_features_without_label(features_df).apply(pd.to_numeric, axis=1, downcast='float')
    else:
        return select_features_without_label(features_df)


def get_dhne_embedding(npy_name, dict_name):
    embeddings = np.load(os.path.join(get_root_path(), "feature_vectors", npy_name))
    replace_dict = load_dict(os.path.join(get_root_path(), "feature_vectors", dict_name))
    reverse_dict = dict(zip(list(replace_dict.keys()), list(range(len(replace_dict.keys())))))
    real_id = OrderedDict({key: list(replace_dict[key].keys()) for key in replace_dict.keys()})
    embedding_df_dict = {i: pd.DataFrame(data=embeddings[reverse_dict[i]], index=real_id[i],
                                         columns=["DHNE" + str(k) for k in
                                                  range(embeddings[reverse_dict[i]].shape[1])])
                         for i in real_id.keys()}
    return embedding_df_dict


# In[4]:


drop_list = ["seller_id", "user_id", "cat_id", "item_id", "brand_id", "user_seller", "seller_brand", "seller_catalog"]

user_info_dict = get_user_info()

user_info_one_hot = pd.get_dummies(user_info_dict, columns=["age_range", "gender"])

all_pairs = get_train_pairs()
test_pairs = get_test_pairs()

all_log = get_train_log(None)
all_log = all_log.loc[(all_log['time_stamp'] == 1111) & (all_log['action_type'] == 2)]  # 选择符合要求的行，这里为选择时间戳为1111的记录
all_log = all_log.drop(['time_stamp', 'action_type'], axis=1)  # 丢掉不需要的列
all_log.fillna(0, inplace=True)

all_log["user_seller"] = np.add(np.array(all_log["user_id"].map(lambda x: str(x) + "_")),
                                np.array(all_log["seller_id"].map(lambda x: str(x))))
all_log["seller_brand"] = np.add(np.array(all_log["seller_id"].map(lambda x: str(x) + "_")),
                                 np.array(all_log["brand_id"].map(lambda x: str(x))))
all_log["seller_catalog"] = np.add(np.array(all_log["seller_id"].map(lambda x: str(x) + "_")),
                                   np.array(all_log["cat_id"].map(lambda x: str(x))))

all_log = pd.merge(all_log, all_pairs, how='left', on=["user_id", "seller_id"], left_on=None, right_on=None,
                   left_index=False, right_index=False, sort=True,
                   suffixes=('_x', '_y'), copy=True, indicator=False,
                   validate=None)

choose_set = set(np.add(np.array(all_pairs["user_id"].map(lambda x: str(x) + "_")),
                        np.array(all_pairs["seller_id"].map(lambda x: str(x))))).union(
    np.add(np.array(test_pairs["user_id"].map(lambda x: str(x) + "_")),
           np.array(test_pairs["seller_id"].map(lambda x: str(x)))))

all_log = all_log[all_log["user_seller"].isin(choose_set)]

all_log = all_log.reset_index(drop=True)

embedding_df_dict_sbc = get_dhne_embedding("embeddings.npy", "train_dict")
embedding_df_dict_us = get_dhne_embedding("us_6.npy", "train_dict_us")
embedding_df_dict_usbc = get_dhne_embedding("usbc_8.npy", "train_dict_usbc")
embedding_df_dict_usb = get_dhne_embedding("usb_8.npy", "train_dict_usb")

cat_v, cat_d, c_r_d = get_catagory_vect_dict()
mer_v, mer_d, m_r_d = get_merchant_vect_dict()
bra_v, bra_d, b_m_d = get_brand_vect_dict()

c_d = get_real_vect_dict(c_r_d, cat_v)
m_d = get_real_vect_dict(m_r_d, mer_v)
b_d = get_real_vect_dict(b_m_d, bra_v)

cat_v, cat_d, c_r_d = get_catagory_vect_buy_dict()
mer_v, mer_d, m_r_d = get_merchant_vect_buy_dict()
bra_v, bra_d, b_m_d = get_brand_vect_buy_dict()

c_d_buy = get_real_vect_dict(c_r_d, cat_v)
m_d_buy = get_real_vect_dict(m_r_d, mer_v)
b_d_buy = get_real_vect_dict(b_m_d, bra_v)

od = operate_days(-1).str.len().to_frame()
p_d = purchase_days(-1).str.len().to_frame()

user_based_feature = load_dict_for_embedding("u_f_d.txt")


# In[5]:


def get_test_set(logs: pd.DataFrame, catalog_feature_file_list=None, merchant_feature_file_list=None,
                 brand_feature_file_list=None,
                 user_feature_file_list=None, merchant_user_feature_file_list=None,
                 merchant_catlog_feature_file_list=None, merchant_brand_feature_file_list=None, cat_f=None, mer_f=None,
                 brd_f=None, usr_f=None, brand_em=None, cat_em=None, seller_em=None, user_em=None):
    if user_em is None:
        user_em = []
    if seller_em is None:
        seller_em = []
    if cat_em is None:
        cat_em = []
    if brand_em is None:
        brand_em = []
    if usr_f is None:
        usr_f = []
    if brd_f is None:
        brd_f = []
    if mer_f is None:
        mer_f = []
    if cat_f is None:
        cat_f = []
    if user_feature_file_list is None:
        user_feature_file_list = []
    if brand_feature_file_list is None:
        brand_feature_file_list = []
    if merchant_feature_file_list is None:
        merchant_feature_file_list = []
    if catalog_feature_file_list is None:
        catalog_feature_file_list = []

    logs = logs[logs["user_seller"].isin(np.add(np.array(test_pairs["user_id"].map(lambda x: str(x) + "_")),
                                                np.array(test_pairs["seller_id"].map(
                                                    lambda x: str(x)))))]

    if len(catalog_feature_file_list) > 0 or len(cat_em) > 0 or len(cat_f) > 0:
        catalog_feature = feature_transform_pipeline(catalog_feature_file_list, cat_em, cat_f)
        logs = logs.join(catalog_feature, on="cat_id", how="left", rsuffix="_c")

    if len(merchant_feature_file_list) > 0 or len(seller_em) > 0 or len(mer_f) > 0:
        merchant_features = feature_transform_pipeline(merchant_feature_file_list, seller_em, mer_f)
        logs = logs.join(merchant_features, on="seller_id", how="left", rsuffix="_m")

    if len(brand_feature_file_list) > 0 or len(brand_em) > 0 or len(brd_f) > 0:
        brand_features = feature_transform_pipeline(brand_feature_file_list, brand_em, brd_f)
        logs = logs.join(brand_features, on="brand_id", how="left", rsuffix="_b")

    if len(user_feature_file_list) > 0 or len(user_em) > 0 or len(usr_f) > 0:
        user_features = feature_transform_pipeline(user_feature_file_list, user_em, usr_f)
        logs = logs.join(user_features, on="user_id", how="left", rsuffix="_u")

    if merchant_user_feature_file_list is not None and len(merchant_user_feature_file_list) > 0:
        merchant_user_features = feature_transform_pipeline(merchant_user_feature_file_list, [], [],
                                                            choose_to_downcast=False)
        logs = logs.join(merchant_user_features, on="user_seller", how="left", rsuffix="_mu")
    if merchant_catlog_feature_file_list is not None and len(merchant_catlog_feature_file_list) > 0:
        merchant_catalog_features = feature_transform_pipeline(merchant_catlog_feature_file_list, [], [],
                                                               choose_to_downcast=False)
        logs = logs.join(merchant_catalog_features, on="seller_catalog", how="left", rsuffix="_mc")
    if merchant_brand_feature_file_list is not None and len(merchant_brand_feature_file_list) > 0:
        merchant_brand_features = feature_transform_pipeline(merchant_brand_feature_file_list, [], [],
                                                             choose_to_downcast=False)
        logs = logs.join(merchant_brand_features, on="seller_brand", how="left", rsuffix="_md")

    test_log = logs[logs["user_seller"].isin(np.add(np.array(test_pairs["user_id"].map(lambda x: str(x) + "_")),
                                                    np.array(test_pairs["seller_id"].map(
                                                        lambda x: str(x)))))].drop_duplicates(
        subset=["user_id", "seller_id"], keep='first')

    test_data = test_log.drop(columns=drop_list + ["label"])
    return test_data, test_log


def predict(test_log, *args):
    dtest = xgb.DMatrix(data=test_log)
    bst = joblib.load(os.path.join(get_root_path(), "model", *args))
    y_test = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    return y_test

# In[7]:


def full():
    feature_list = dict()

    feature_list["merchant_feature_file_list"] = [
        "seller_time.csv",
        "seller_buy_repeat_no.csv",
        "seller_buy_unique.csv",
        "seller_click_time.csv"
    ]

    feature_list["catalog_feature_file_list"] = [
        "catalogs_time.csv",
        "catlogs_buy_repeat_no.csv",
    ]

    feature_list["brand_feature_file_list"] = [
        "brands_time.csv",
        "brands_buy_repeat_no.csv"
    ]

    feature_list["user_feature_file_list"] = ["users_time.csv",
                                              "users_click_time.csv",
                                              ]

    feature_list["merchant_user_feature_file_list"] = [
        "seller_user_click_time.csv",
        "seller_user_buy_time.csv"
    ]
    feature_list["merchant_catlog_feature_file_list"] = [
        "seller_cat_buy_unique.csv",
    ]
    #
    feature_list["merchant_brand_feature_file_list"] = [
        "seller_brand_buy_unique.csv"  # good
    ]

    feature_list["brand_em"] = [
        b_d,
        b_d_buy]
    feature_list["cat_em"] = [
        c_d,
        c_d_buy
    ]

    feature_list["seller_em"] = [
        m_d,
        m_d_buy
    ]
    feature_list["user_em"] = [
        user_based_feature
    ]
    feature_list["usr_f"] = [
        p_d,
        od,

    ]

    test_data, test_log = get_test_set(all_log, **feature_list)

    model_name_list = get_file_list_in_dir(os.path.join(get_root_path(), "model", "12.24"))

    for name in model_name_list:
        y_test = predict(test_data, name)
        test_log["prob"] = y_test
        test_log.groupby(["user_id", "seller_id"])["prob"].mean().reset_index().to_csv(
            os.path.join(get_root_path(), "prediction", "12.24", "prob_x" + name + ".csv"), encoding='utf-8',
            index=False)


full()
