# coding: utf-8

import matplotlib
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from deep_learning.data_convert import *
from sklearn import metrics
import xgboost as xgb
import matplotlib.pyplot as plt
from basic_model.feature_extraction.user_profile import operate_days, purchase_days
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from feature_selector import FeatureSelector
import lightgbm as lgb
import os


# In[2]:


def read_feature_file(file_name) -> pd.DataFrame:
    print("Reading:", file_name)
    file_path = os.path.join(get_root_path(), "feature_vectors", file_name)
    return pd.read_csv(file_path, index_col=0)


# In[3]:


def select_features_without_label(features: pd.DataFrame, missing_threshold=0.7,
                                  correlation_threshold=0.95) -> pd.DataFrame:
    fs = FeatureSelector(data=features)
    fs.identify_missing(missing_threshold)
    fs.identify_single_unique()
    #     fs.identify_collinear(correlation_threshold)
    return fs.remove(methods=['missing', 'single_unique'])


# In[4]:


def read_all_features(file_name_list: list) -> list:
    return [select_features_without_label(read_feature_file(file_name)) for file_name in file_name_list]


# In[5]:


def merge_all_features_on_index(feature_df_list: list) -> pd.DataFrame:
    first_one = feature_df_list.pop()
    time_suffix = 0
    for feature_df in feature_df_list:
        time_suffix += 1
        first_one = first_one.join(feature_df, how="outer", rsuffix="_" + str(time_suffix))
    return first_one


# In[6]:


def feature_transform_pipeline(file_name_list: list, dic_list=[], df_list=[]) -> pd.DataFrame:
    features_df_list = read_all_features(file_name_list) + [select_features_without_label(pd.DataFrame.from_dict(i).T)
                                                            for i in dic_list] + df_list
    features_df = merge_all_features_on_index(features_df_list)
    return select_features_without_label(features_df)


# In[7]:


drop_list = ["seller_id", "user_id", "cat_id", "item_id", "brand_id", "user_seller", "seller_brand", "seller_catalog"]

# ## Prepare data

# In[8]:


user_info_dict = get_user_info()

# In[9]:


user_info_one_hot = pd.get_dummies(user_info_dict, columns=["age_range", "gender"])

# In[10]:


all_pairs = get_train_pairs()
test_pairs = get_test_pairs()

# In[11]:


all_log = get_train_log(None)
all_log = all_log.loc[(all_log['time_stamp'] == 1111) & (all_log['action_type'] == 2)]  # 选择符合要求的行，这里为选择时间戳为1111的记录
all_log = all_log.drop(['time_stamp', 'action_type'], axis=1)  # 丢掉不需要的列
all_log.fillna(0, inplace=True)

# In[12]:


all_log["user_seller"] = np.add(np.array(all_log["user_id"].map(lambda x: str(x) + "_")),
                                np.array(all_log["seller_id"].map(lambda x: str(x))))
all_log["seller_brand"] = np.add(np.array(all_log["seller_id"].map(lambda x: str(x) + "_")),
                                 np.array(all_log["brand_id"].map(lambda x: str(x))))
all_log["seller_catalog"] = np.add(np.array(all_log["seller_id"].map(lambda x: str(x) + "_")),
                                   np.array(all_log["cat_id"].map(lambda x: str(x))))

# In[13]:


all_log = pd.merge(all_log, all_pairs, how='left', on=["user_id", "seller_id"], left_on=None, right_on=None,
                   left_index=False, right_index=False, sort=True,
                   suffixes=('_x', '_y'), copy=True, indicator=False,
                   validate=None)

# In[14]:


choose_set = set(np.add(np.array(all_pairs["user_id"].map(lambda x: str(x) + "_")),
                        np.array(all_pairs["seller_id"].map(lambda x: str(x))))).union(
    np.add(np.array(test_pairs["user_id"].map(lambda x: str(x) + "_")),
           np.array(test_pairs["seller_id"].map(lambda x: str(x)))))

# In[15]:


all_log = all_log[all_log["user_seller"].isin(choose_set)]

# In[16]:


all_log = all_log.reset_index(drop=True)

# ## Read and select features

# In[17]:


merchant_feature_file_list = ["seller_buy_repeat_no.csv",
                              "seller_buy_unique.csv",
                              "seller_time.csv",
                              "seller_click_time_user.csv",
                              "seller_buy_reapeat_deep.csv"
                              ]

# In[18]:


catlog_feature_file_list = ["catalogs_time.csv",
                            "catlogs_buy_repeat_no.csv",
                            "catalogs_buy_time_deep.csv"
                            ]

# In[19]:


brand_feature_file_list = ["brands_time.csv",
                           "brands_buy_repeat_no.csv"
                           ]

# In[20]:


user_feature_file_list = ["users_time.csv",
                          "users_buy_repeat_no_month.csv",
                          "users_click_time.csv",
                          "users_click_repeat_deep.csv",
                          "users_buy_repeat_deep.csv",
                          "users_buy_unique.csv"
                          ]

# In[21]:


merchant_user_feature_file_list = ["seller_user_click_time.csv",
                                   "seller_user_buy_time.csv"
                                   ]

# In[22]:


merchant_catlog_feature_file_list = ["seller_cat_buy_unique.csv",
                                     "seller_cat_buy_reapeat.csv"]

# In[23]:


merchant_brand_feature_file_list = ["seller_brand_buy_unique.csv",
                                    "seller_brand_buy_repeat.csv"]

# In[24]:


embeddings = np.load(os.path.join(get_root_path(), "feature_vectors", "embeddings.npy"))
replace_dict = load_dict(os.path.join(get_root_path(), "one_time_use", "train_dict"))
reverse_dict = dict(zip(list(replace_dict.keys()), list(range(len(replace_dict.keys())))))
real_id = OrderedDict({key: list(replace_dict[key].keys()) for key in replace_dict.keys()})
embedding_df_dict = {i: pd.DataFrame(data=embeddings[reverse_dict[i]], index=real_id[i],
                                     columns=["DHNE_sbc" + str(k) for k in range(embeddings[reverse_dict[i]].shape[1])]) for
                     i in real_id.keys()}

embeddings = np.load(os.path.join(get_root_path(), "feature_vectors", "usb_8.npy"))
replace_dict = load_dict(os.path.join(get_root_path(), "feature_vectors", "train_dict_usb"))
reverse_dict = dict(zip(list(replace_dict.keys()), list(range(len(replace_dict.keys())))))
real_id = OrderedDict({key: list(replace_dict[key].keys()) for key in replace_dict.keys()})
embedding_df_dict_usb = {i: pd.DataFrame(data=embeddings[reverse_dict[i]], index=real_id[i],
                                         columns=["DHNE_usb" + str(k) for k in range(embeddings[reverse_dict[i]].shape[1])])
                         for i in real_id.keys()}

# In[25]:


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

# In[26]:


od = operate_days(-1).str.len().to_frame()
p_d = purchase_days(-1).str.len().to_frame()

# In[27]:


user_based_feature = load_dict_for_embedding("u_f_d.txt")

# In[28]:


catlog_feature = feature_transform_pipeline(catlog_feature_file_list, [c_d, c_d_buy], [embedding_df_dict["cat_id"]])
merchant_features = feature_transform_pipeline(merchant_feature_file_list, [m_d, m_d_buy],
                                               [embedding_df_dict["seller_id"], embedding_df_dict_usb["seller_id"]])
brand_features = feature_transform_pipeline(brand_feature_file_list, [b_d, b_d_buy],
                                            [embedding_df_dict["brand_id"], embedding_df_dict_usb["brand_id"]])
user_features = feature_transform_pipeline(user_feature_file_list, [user_based_feature],
                                           [od, p_d, embedding_df_dict_usb["user_id"]])
merchant_user_features = feature_transform_pipeline(merchant_user_feature_file_list)
merchant_catalog_features = feature_transform_pipeline(merchant_catlog_feature_file_list)
merchant_brand_features = feature_transform_pipeline(merchant_brand_feature_file_list)

# ## Merge features

# In[29]:


all_log = all_log.join(catlog_feature, on="cat_id", how="left", rsuffix="_c")
all_log = all_log.join(brand_features, on="brand_id", how="left", rsuffix="_b")
all_log = all_log.join(user_features, on="user_id", how="left", rsuffix="_u")
all_log = all_log.join(merchant_features, on="seller_id", how="left", rsuffix="_m")
all_log = all_log.join(merchant_user_features, on="user_seller", how="left", rsuffix="_mu")
all_log = all_log.join(merchant_catalog_features, on="seller_catalog", how="left", rsuffix="_mc")
all_log = all_log.join(merchant_brand_features, on="seller_brand", how="left", rsuffix="_md")

# In[30]:


train_log = all_log[all_log["user_seller"].isin(np.add(np.array(all_pairs["user_id"].map(lambda x: str(x) + "_")),
                                                       np.array(all_pairs["seller_id"].map(
                                                           lambda x: str(x)))))].drop_duplicates(
    subset=["user_id", "seller_id"], keep='first')
test_log = all_log[all_log["user_seller"].isin(np.add(np.array(test_pairs["user_id"].map(lambda x: str(x) + "_")),
                                                      np.array(test_pairs["seller_id"].map(
                                                          lambda x: str(x)))))].drop_duplicates(
    subset=["user_id", "seller_id"], keep='first')

# In[39]:


msk = np.random.rand(len(all_pairs)) < 0.9
train_pair = all_pairs[msk]
valid_pair = all_pairs[~msk]

# In[40]:


train_log = all_log[all_log["user_seller"].isin(np.add(np.array(train_pair["user_id"].map(lambda x: str(x) + "_")),
                                                       np.array(train_pair["seller_id"].map(lambda x: str(x)))))]
valid_log = all_log[all_log["user_seller"].isin(np.add(np.array(valid_pair["user_id"].map(lambda x: str(x) + "_")),
                                                       np.array(valid_pair["seller_id"].map(lambda x: str(x)))))]

# ## RandomSearch

# In[ ]:


import scipy.stats as st

one_to_left = st.beta(10, 1)
from_zero_positive = st.expon(0, 50)


params = {
    "n_estimators": st.randint(20, 200),
    "max_depth": st.randint(1, 10),
    "learning_rate": st.uniform(0.005, 0.5),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 20),
    'reg_alpha': from_zero_positive,
    "min_child_weight": st.uniform(0, 500),
}

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

xgbcla = XGBClassifier(booster='gbtree', objective='binary:logistic', eval_metric='auc', silent=1, n_jobs=1)

train_no_duplicate = train_log

gs = RandomizedSearchCV(xgbcla, params, n_jobs=1, cv=5,
                        scoring="roc_auc")
gs.fit(train_no_duplicate.drop(columns=["label"] + drop_list).values, train_no_duplicate["label"])

print(gs.best_params_)
try:
    print(gs.cv_results_)
except:
    print("error")

# In[ ]:


print(gs.best_score_)

# In[ ]:

try:
    ytest = gs.predict_proba(test_log.drop(columns=drop_list + ["label"]).values)

    # In[ ]:

    test_log["prob"] = ytest.transpose()[1]

    # In[ ]:

    test_log.groupby(["user_id", "seller_id"])["prob"].mean().reset_index().to_csv(
        os.path.join(get_root_path(), "prob_b.csv"), encoding='utf-8', index=False)
except:
    import logging

    print(logging.exception("message"))

with open("best_score.txt", "w") as f:
    f.write(str(gs.best_score_))
with open("best_params.txt", "w") as f:
    f.write(str(gs.best_params_))
with open("best_cv_result.txt", "w") as f:
    f.write(str(gs.cv_results_))
