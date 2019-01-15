import os

import numpy as np
import pandas as pd


def get_root_path() -> str:
    """

    :return:
    """
    return os.path.abspath(os.path.dirname(__file__))


def fix_nan(df, fill_num=1):
    """
    对传进来的DataFrame中的空值填充指定数字
    :param df: DataFrame
    :return:
    """
    return df.fillna(fill_num, inplace=True)


def get_user_info():
    """
    获取年龄等用户信息
    :return:
    """
    my_path = os.path.abspath(os.path.dirname(__file__))  # avoid import error
    df = pd.read_csv(my_path + './data/data_format1/user_info_format1.csv')
    fix_nan(df['gender'], 2)
    fix_nan(df['age_range'], 0)
    df['age_range'] = df['age_range'].astype(np.int32)
    df['gender'] = df['gender'].astype(np.int32)
    return df


def get_train_log(row_number=500000) -> pd.DataFrame:
    """
    返回log文件，
    :param row_number: 需要读取的行数
    :return: 返回DataFrame格式
    """
    my_path = os.path.abspath(os.path.dirname(__file__))  # avoid import error
    df = pd.read_csv(my_path + './data/data_format1/user_log_format1.csv', nrows=row_number)
    fix_nan(df)
    df["brand_id"] = df["brand_id"].astype(np.int32)
    df["time_stamp"] = df["time_stamp"].astype(np.int32)
    return df


def get_train_pairs() -> pd.DataFrame:
    """
    读取有标签的训练数据
    :return:DataFrame
    """
    my_path = os.path.abspath(os.path.dirname(__file__))  # avoid import error
    df = pd.read_csv(my_path + "./data/train_format1.csv")
    df = df.rename(columns={'merchant_id': 'seller_id'})
    return df


def read_dat_file(path: str) -> dict:
    """

    :param path:
    :return:
    """
    my_path = os.path.abspath(os.path.dirname(__file__))  # avoid import error
    feature_dict = {}
    with open(my_path + path, "r") as f:
        lines = f.read().split("\n")
        lines.pop(-1)
        for line in lines:
            words = line.split(' ')
            feature_dict[int(words[0][1:])] = [float(num) for num in words[1:-1]]
    return feature_dict


def get_test_pairs():
    """
    读取无标签的提交数据
    :return:
    """
    my_path = os.path.abspath(os.path.dirname(__file__))  # avoid import error
    df = pd.read_csv(my_path + "./data/test_format1.csv")
    df = df.rename(columns={'merchant_id': 'seller_id'})
    return df


def save_dict(dict, path):
    """
    保存字典
    :param dict:
    :param path:
    :return:
    """
    with open(path, "w") as f:
        f.write(str(dict))
        return


def load_dict_for_embedding(path):
    """
    从指定目录读取字典
    :param path:
    :return:
    """
    with open(os.path.join(get_root_path(), "feature_vectors", path), "r") as f:
        dic = f.read()
    return eval(dic)


def load_dict(path):
    """
    从指定目录读取字典
    :param path:
    :return:
    """
    with open(path, "r") as f:
        dic = f.read()
    return eval(dic)


def choose_logs_in_train_and_test(logs: pd.DataFrame, entity="user_seller") -> pd.DataFrame:
    """
    Used by feature extraction
    :param entity: The id which determine what to keep
    :param logs:
    :return:
    """
    train_pairs = get_test_pairs()
    test_pairs = get_test_pairs()
    full_pairs = pd.concat([test_pairs, train_pairs])
    if entity == "user_seller":
        logs["user_seller"] = np.add(np.array(logs["user_id"].map(lambda x: str(x) + "_")),
                                     np.array(logs["seller_id"].map(lambda x: str(x))))
        choose_set = set(np.add(np.array(full_pairs["user_id"].map(lambda x: str(x) + "_")),
                                np.array(full_pairs["seller_id"].map(lambda x: str(x)))))
        return logs[logs["user_seller"].isin(choose_set)]
    else:
        choose_set = set(full_pairs[entity])
        return logs[logs[entity].isin(choose_set)]


def get_file_list_in_dir(path):
    return [name for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name))]


def time_cost(func):
    """
    @time_cost  放在任意函数定义的上一行，运行时便能自动打印出函数所用时间
    :param func:
    :return:
    """

    def wrapper(*args, **kw):
        import time
        start_time = time.time()
        result = func(*args, **kw)
        finish_time = time.time()
        print("used_time(s):", finish_time - start_time)
        return result

    return wrapper


if __name__ == "__main__":
    # load_dict_for_embedding("./breand_2_vec.txt")
    print(get_file_list_in_dir("./"))
