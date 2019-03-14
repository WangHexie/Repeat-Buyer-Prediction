from typing import List

from basic_function import *


class TempVariable:
    def __init__(self, agg_id=None):
        self.__agg_id = agg_id

    def old_x_count_on_de(self, x: pd.DataFrame) -> pd.Series:
        names = {
            "old_buy_count_on_de": self.__old_buy_count_on_de_func(x),
            "all_x_on_de": len(set(x[x["time_stamp"] == 1111][self.__agg_id]))
        }

        return pd.Series(names, index=["old_buy_count_on_de", "all_x_on_de"])

    def more_than_once_ratio(self, x: pd.DataFrame) -> pd.Series:
        names = {
            "more_than_once_number": int(sum((x.groupby([self.__agg_id]).count() > 1)["item_id"])),
            "all_x_count": len(set(x[self.__agg_id]))
        }

        return pd.Series(names, index=["more_than_once_number", "all_x_count"])

    def __old_buy_count_on_de_func(self, x: pd.DataFrame):
        buy_on_de_list = set(x[x["time_stamp"] == 1111][self.__agg_id])
        buy_not_on_de_list = set(x[x["time_stamp"] != 1111][self.__agg_id])

        old_buy_count_on_de = sum(i in buy_not_on_de_list for i in buy_on_de_list)
        return old_buy_count_on_de


def agg_decoration(feature_name):
    def base_feature(x: pd.DataFrame) -> pd.Series:
        names = {
            feature_name: x["item_id"].count()
        }

        return pd.Series(names, index=[feature_name])

    return base_feature


def merge_all(df_list):
    return pd.concat(df_list, axis=1, sort=False).fillna(0)


def basic_ratio(feature: pd.DataFrame):
    feature["buy_ratio"] = feature["buy_count"] / feature["all_count"]
    feature["click_ratio"] = feature["click_count"] / feature["all_count"]
    feature["cart_ratio"] = feature["cart_count"] / feature["all_count"]
    feature["like_ratio"] = feature["like_count"] / feature["all_count"]
    feature["de_buy_ratio"] = feature["de_buy_count"] / feature["all_count"]
    for action_type in range(3):
        feature[str(action_type) + "_" + "un_de_ratio"] = feature[str(action_type) + "_" + "un_de_count"] / feature[
            "un_de_all_count"]
    return feature


def basic_count(x: pd.DataFrame, agg_id) -> List[pd.DataFrame,]:
    final_list = [
        x.groupby([agg_id]).apply(agg_decoration("all_count")),
        x[x["time_stamp"] != 1111].groupby([agg_id]).apply(agg_decoration("un_de_all_count")),
        x[x["action_type"] == 1].groupby([agg_id]).apply(agg_decoration("cart_count")),
        x[x["action_type"] == 0].groupby([agg_id]).apply(agg_decoration("click_count")),
        x[x["action_type"] == 2].groupby([agg_id]).apply(agg_decoration("buy_count")),
        x[x["action_type"] == 3].groupby([agg_id]).apply(agg_decoration("like_count")),
        x[(x["action_type"] == 2) & (x["time_stamp"] == 1111)].groupby([agg_id]).apply(
            agg_decoration("de_buy_count")),
    ]

    action_un_de = [
        x[x["action_type"] == action_type].groupby([agg_id]).apply(
            agg_decoration(str(action_type) + "_" + "un_de_count")) for action_type in range(3)
    ]

    return final_list + action_un_de


# user_feature
logs = get_train_log(None)
logs = choose_logs_in_train_and_test(logs, entity="user_id")
user_feature_list = basic_count(logs, "user_id") + [
    logs[logs["action_type"] == 2].groupby(["user_id"]).apply(TempVariable(agg_id="seller_id").old_x_count_on_de),
    logs[logs["action_type"] == 2].groupby(["user_id"]).apply(TempVariable(agg_id="seller_id").more_than_once_ratio),
]

user_feature_df = merge_all(user_feature_list)
user_feature_df = basic_ratio(user_feature_df)
user_feature_df["old_x_buy_on_de"] = user_feature_df["old_buy_count_on_de"] / user_feature_df["all_x_on_de"]
user_feature_df["more_than_once_ratio"] = user_feature_df["more_than_once_number"] / user_feature_df["all_x_count"]
user_feature_df.to_csv(os.path.join(get_root_path(), "feature_vectors",
                                    "user_ratio"), float_format='%.4f', index_label="index")


# user_merchant_feature
logs = get_train_log(None)
logs["user_seller"] = np.add(np.array(logs["user_id"].map(lambda x: str(x) + "_")),
                             np.array(logs["seller_id"].map(lambda x: str(x))))
logs = choose_logs_in_train_and_test(logs, entity="user_seller")
user_merchant_feature_list = basic_count(logs, "user_seller")
user_merchant_feature_df = merge_all(user_merchant_feature_list)
user_merchant_feature_df = basic_ratio(user_merchant_feature_df)
user_merchant_feature_df.to_csv(os.path.join(get_root_path(), "feature_vectors",
                                             "user_seller_ratio"), float_format='%.4f', index_label="index")

# merchant_feature
logs = get_train_log(None)
logs = choose_logs_in_train_and_test(logs, entity="seller_id")
merchant_feature_list = basic_count(logs, "seller_id") + [
    logs[logs["action_type"] == 2].groupby(["seller_id"]).apply(TempVariable(agg_id="user_id").old_x_count_on_de),
    logs[logs["action_type"] == 2].groupby(["seller_id"]).apply(TempVariable(agg_id="user_id").more_than_once_ratio)
]

merchant_feature_df = merge_all(merchant_feature_list)
merchant_feature_df = basic_ratio(merchant_feature_df)
merchant_feature_df["old_x_buy_on_de"] = merchant_feature_df["old_buy_count_on_de"] / merchant_feature_df["all_x_on_de"]
merchant_feature_df["more_than_once_ratio"] = merchant_feature_df["more_than_once_number"] / merchant_feature_df[
    "all_x_count"]
merchant_feature_df.to_csv(os.path.join(get_root_path(), "feature_vectors",
                                        "seller_ratio"), float_format='%.4f', index_label="index")
