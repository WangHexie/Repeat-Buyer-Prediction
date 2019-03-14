import random
from typing import List

from pandas import DataFrame

from deep_learning.data_convert import *
from feature_selector import FeatureSelector


def read_feature_file(file_name) -> pd.DataFrame:
    print("Reading:", file_name)
    file_path = os.path.join(get_root_path(), "feature_vectors", file_name)
    df = pd.read_csv(file_path, index_col=0)
    df.columns = df.columns.map(lambda x: str(x) + file_name)
    print("feature length:", len(df.columns.values))
    return df


def select_features_without_label(features: pd.DataFrame, missing_threshold=0.99,
                                  correlation_threshold=1.0) -> pd.DataFrame:
    fs = FeatureSelector(data=features)
    fs.identify_missing(missing_threshold)
    fs.identify_single_unique()
    if correlation_threshold < 1:
        fs.identify_collinear(correlation_threshold)
        return fs.remove(methods=['missing', 'single_unique', "collinear"])
    else:
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


def dict_to_df(feature: dict):
    f_df = pd.DataFrame.from_dict(feature).T
    f_df.columns = f_df.columns.map(lambda x: str(x) + "_dic_" + str(random.random()))
    return f_df


def feature_transform_pipeline(file_name_list: list, dic_list: list = None, df_list: list = None,
                               choose_to_downcast=True) -> pd.DataFrame:
    features_df_list: List[DataFrame] = read_all_features(file_name_list) + ([
                                                                                 select_features_without_label(
                                                                                     dict_to_df(i))
                                                                                 for i in
                                                                                 dic_list] if dic_list is not None else []) + (
                                            [
                                                select_features_without_label(i) for i
                                                in df_list] if df_list is not None else [])
    features_df = merge_all_features_on_index(features_df_list)
    if choose_to_downcast:
        return select_features_without_label(features_df).apply(pd.to_numeric, axis=1, downcast='float')
    else:
        return select_features_without_label(features_df)


def get_dhne_embedding(npy_name, dict_name):
    embeddings = np.load(os.path.join(get_root_path(), "feature_vectors", npy_name))
    replace_dict = load_dict(os.path.join(get_root_path(), "feature_vectors", dict_name))
    shapes = []
    for i in range(embeddings.shape[0]):
        shapes.append(embeddings[i].shape[0])
    correspond_index = []
    for i in replace_dict.keys():
        correspond_index.append(shapes.index(len(replace_dict[i].keys())))
    reverse_dict = dict(zip(list(replace_dict.keys()), correspond_index))
    real_id = OrderedDict({key: list(replace_dict[key].keys()) for key in replace_dict.keys()})
    embedding_df_dict = {i: pd.DataFrame(data=embeddings[reverse_dict[i]], index=real_id[i],
                                         columns=["DHNE" + str(k) for k in
                                                  range(embeddings[reverse_dict[i]].shape[1])])
                         for i in real_id.keys()}
    return embedding_df_dict


def get_w2v_embedding(npy_name, dict_name):
    embeddings = np.load(os.path.join(get_root_path(), "feature_vectors", npy_name))
    replace_dict = load_dict(os.path.join(get_root_path(), "feature_vectors", dict_name))
    # replace_dict.pop("user_id", None)
    # reverse_dict = dict(zip(list(replace_dict.keys()), [0, 2, 1]))
    replace_dict = load_dict(os.path.join(get_root_path(), "feature_vectors", dict_name))
    shapes = []
    for i in range(embeddings.shape[0]):
        shapes.append(embeddings[i].shape[0])
    correspond_index = []
    for i in replace_dict.keys():
        correspond_index.append(shapes.index(len(replace_dict[i].keys())))
    real_id = OrderedDict({key: list(replace_dict[key].keys()) for key in replace_dict.keys()})
    embedding_df_dict = {id_name: pd.DataFrame(data=embeddings[reverse_dict[id_name]], index=real_id[id_name],
                                               columns=["DHNE" + str(k) for k in
                                                        range(embeddings[reverse_dict[id_name]].shape[1])])
                         for id_name in real_id.keys()}
    return embedding_df_dict


def get_w2v_attr_embedding(npy_name, dict_name):
    embeddings = np.load(os.path.join(get_root_path(), "feature_vectors", npy_name))
    replace_dict = load_dict(os.path.join(get_root_path(), "feature_vectors", dict_name))
    real_id = OrderedDict({key: list(replace_dict[key].keys()) for key in replace_dict.keys()})
    embedding_df_dict = {
        "user_id": pd.DataFrame(data=embeddings[2], index=real_id["user_id"],
                                columns=["DHNE_id" + str(k) for k in
                                         range(embeddings[0].shape[1])]),
        "age": pd.DataFrame(data=embeddings[0], index=list(range(9)),
                            columns=["DHNE_age" + str(k) for k in
                                     range(embeddings[0].shape[1])]),
        "gender": pd.DataFrame(data=embeddings[1], index=list(range(3)),
                               columns=["DHNE_gender" + str(k) for k in
                                        range(embeddings[0].shape[1])])
    }
    return embedding_df_dict


if __name__ == '__main__':
    embedding_df_dict_sbc = get_dhne_embedding("usb_128_5.npy", "train_dict_usb_s")
    # feature_transform_pipeline([
    #     "seller_user_click_time.csv"
    #     # "seller_user_buy_time.csv"
    # ])
    # embedding_df_dict_usb = get_dhne_embedding("usb_s_16.npy", "train_dict_usb_s")
    # print(get_w2v_attr_embedding("attr_embedding_w2v.npy", "id_dict_w2v.txt"))
    # embedding_df_dict_sbc = get_w2v_embedding("embedding_w2v.npy", "id_dict_w2v.txt")
