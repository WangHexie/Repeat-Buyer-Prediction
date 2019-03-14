# coding: utf-8

# In[1]:


import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from basic_function import get_train_log, choose_logs_in_train_and_test, get_root_path
from sklearn.decomposition import NMF


def get_buy_list():
    name_list = ["user_id", "seller_id"]

    logs = get_train_log(None)
    logs = logs[logs["action_type"] == 2][["user_id", "seller_id"]]
    replace = OrderedDict({name_list[i]: OrderedDict(
        zip(set(logs.loc[:, name_list[i]].values), list(range(len(set(logs.loc[:, name_list[i]].values)))))) for i in
    range(len(name_list))})
    for i in range(len(name_list)):
        logs[name_list[i]] = logs[name_list[i]].map(replace[name_list[i]])

    a = logs.groupby(["user_id", "seller_id"]).size().reset_index(name='counts')
    data = a.values
    shape = (len(set(logs["user_id"])), len(set(logs["seller_id"])))
    matrix = np.zeros(shape, dtype=np.int8)

    for user, item, rating in data:
        matrix[user][item] = rating  # Convert to 0-based index

    return pd.DataFrame(data=matrix, index=replace["user_id"].keys(), columns=range(matrix.shape[1]))
