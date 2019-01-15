import os

import numpy as np
import pandas as pd

from basic_function import get_file_list_in_dir, get_root_path


def sigmoid_ver(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def blend(file_name_list, time_str, sum=False):
    prob = [pd.read_csv(os.path.join(get_root_path(), "prediction", time_str, i)).sort_values(by="seller_id", axis=0,
                                                                                              kind="mergesort").sort_values(
        by="user_id", axis=0, kind="mergesort")['prob'] for i in file_name_list]
    if sum:
        exp = [i for i in prob]
        final_prob = np.sum(exp, axis=0) / len(exp)
    else:
        exp = [sigmoid_ver(i) for i in prob]
        final_prob = sigmoid(np.sum(exp, axis=0) / len(exp))
    us_df = pd.read_csv(os.path.join(get_root_path(), "prediction", time_str, file_name_list[0])).sort_values(
        by="seller_id", axis=0, kind="mergesort").sort_values(
        by="user_id",
        axis=0,
        kind="mergesort")
    us_df['prob'] = final_prob
    us_df.to_csv(os.path.join(get_root_path(), "prediction", time_str, "blending.csv"), index=False,
                 float_format='%.16f')


if __name__ == '__main__':
    name = "new"
    blend(file_name_list=get_file_list_in_dir(os.path.join(get_root_path(), "prediction", name)), time_str=name,
          sum=False)
