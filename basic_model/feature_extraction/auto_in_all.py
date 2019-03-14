from typing import List

import featuretools as ft
from featuretools.primitives import TimeSincePrevious
from featuretools.primitives import NUnique

from data_munipulation.get_formatted_data import *


def extraction(entity: str, action_type: List[int], name_to_save: str, interesting_value: dict, agg_pre: list,
               depth: int, variable_type: dict = None, drop_list: list = [], sub_entity_list: list = [],
               trans_pre: list=[]):
    log_df = get_train_log(None)
    # choose action type which used
    log_df = log_df.loc[log_df['action_type'].isin(action_type)]

    # choose logs by entity used
    log_df = choose_logs_in_train_and_test(log_df, entity=entity)
    log_df = log_df.reset_index(drop=True)
    log_df["index"] = log_df.index  # required by featuretools

    log_df["month"] = log_df["time_stamp"].map(lambda x: int(x / 100))
    log_df['data'] = log_df["time_stamp"].map(lambda x: '2016-' + str(int(x / 100)) + '-' + str(int(x // 100)))
    user_df = get_user_info()
    log_df = log_df.merge(user_df, on="user_id", how="inner")
    log_df["before_pro"] = log_df["time_stamp"].map(lambda x: (1101 < x) and (x < 1111))

    # drop useless column
    log_df.drop(labels=drop_list, axis=1, inplace=True)

    es = ft.EntitySet(id="logs")

    # select feature column
    if entity == "user_id":
        log_df.drop(labels=["gender", "age_range"], axis=1, inplace=True)
        es = es.entity_from_dataframe(entity_id="logs",
                                      dataframe=log_df,
                                      index="index",
                                      time_index="data",
                                      variable_types=variable_type if variable_type is not None else {
                                          "user_id": ft.variable_types.Categorical,
                                          "item_id": ft.variable_types.Categorical,
                                          "cat_id": ft.variable_types.Categorical,
                                          "seller_id": ft.variable_types.Categorical,
                                          "brand_id": ft.variable_types.Categorical,
                                          "month": ft.variable_types.Categorical,
                                          "time_stamp": ft.variable_types.Categorical,
                                          "data": ft.variable_types.Datetime,
                                          'action_type': ft.variable_types.Categorical,
                                          "before_pro": ft.variable_types.Boolean,
                                      }
                                      )
        es = es.normalize_entity(base_entity_id="logs",
                                 new_entity_id="user_id",
                                 index="user_id")
        es = es.normalize_entity(base_entity_id="logs",
                                 new_entity_id="seller_id",
                                 index="seller_id")
    elif entity == "user_seller":
        log_df["user_seller"] = np.add(np.array(log_df["user_id"].map(lambda x: str(x) + "_")),
                                       np.array(log_df["seller_id"].map(lambda x: str(x))))
        log_df.drop(labels=['user_id', 'seller_id', 'age_range', 'gender'], axis=1, inplace=True)
        es = es.entity_from_dataframe(entity_id="logs",
                                      dataframe=log_df,
                                      index="index",
                                      time_index="data",
                                      variable_types=variable_type if variable_type is not None else {
                                          "item_id": ft.variable_types.Categorical,
                                          "cat_id": ft.variable_types.Categorical,
                                          "brand_id": ft.variable_types.Categorical,
                                          "data": ft.variable_types.Datetime,
                                          "user_seller": ft.variable_types.Categorical,
                                          "time_stamp": ft.variable_types.Categorical,
                                          'action_type': ft.variable_types.Categorical,
                                          "month": ft.variable_types.Categorical,
                                          "before_pro": ft.variable_types.Boolean

                                      }
                                      )
        es = es.normalize_entity(base_entity_id="logs",
                                 new_entity_id="user_seller",
                                 index="user_seller")
    elif entity == "seller_id":
        es = es.entity_from_dataframe(entity_id="logs",
                                      dataframe=log_df,
                                      index="index",
                                      time_index="data",
                                      variable_types=variable_type if variable_type is not None else {
                                          "user_id": ft.variable_types.Categorical,
                                          "item_id": ft.variable_types.Categorical,
                                          "cat_id": ft.variable_types.Categorical,
                                          "seller_id": ft.variable_types.Categorical,
                                          "brand_id": ft.variable_types.Categorical,
                                          "time_stamp": ft.variable_types.Categorical,
                                          "month": ft.variable_types.Categorical,
                                          'action_type': ft.variable_types.Categorical,
                                          "before_pro": ft.variable_types.Boolean,
                                          "age_range": ft.variable_types.Categorical,
                                          "gender": ft.variable_types.Categorical

                                      }
                                      )
        if "user_id" not in drop_list:
            es = es.normalize_entity(base_entity_id="logs",
                                     new_entity_id="user_id",
                                     index="user_id")
        es = es.normalize_entity(base_entity_id="logs",
                                 new_entity_id="seller_id",
                                 index="seller_id")

    for key in interesting_value.keys():
        es["logs"][key].interesting_values = interesting_value[key]

    for sub_entity in sub_entity_list:
        es = es.normalize_entity(base_entity_id="logs",
                                 new_entity_id=sub_entity,
                                 index=sub_entity)

    print("start")
    feature_defs = ft.dfs(entityset=es,
                          target_entity=entity,
                          agg_primitives=agg_pre,
                          max_depth=depth,
                          where_primitives=agg_pre,
                          trans_primitives=trans_pre,
                          features_only=True
                          )
    print(feature_defs)

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity=entity,
                                          agg_primitives=agg_pre,
                                          max_depth=depth,
                                          where_primitives=agg_pre,
                                          trans_primitives=trans_pre,
                                          n_jobs=1,
                                          verbose=True
                                          )

    print(feature_defs)
    feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)
    print(features_enc)
    feature_matrix_enc.to_csv(os.path.join(get_root_path(), "feature_vectors",
                                           name_to_save), float_format='%.4f', index_label="index")


if __name__ == '__main__':
    seller_buy_mode = {
        "entity": "seller_id",
        "action_type": [2],
        "name_to_save": "seller_buy_mode.csv",
        "interesting_value": {"before_pro": [True, False]},
        "agg_pre": ["mode"],
        "depth": 3,
        "variable_type": {
            "seller_id": ft.variable_types.Categorical,
            "month": ft.variable_types.Categorical,
            'action_type': ft.variable_types.Categorical,
            "before_pro": ft.variable_types.Boolean,
            "age_range": ft.variable_types.Categorical,
            "gender": ft.variable_types.Categorical
        },
        "drop_list": ["user_id", "item_id", "cat_id", "brand_id", "time_stamp"]
    }

    seller_fav_count = {
        "entity": "seller_id",
        "action_type": [1, 3],
        "name_to_save": "seller_fav_count.csv",
        "interesting_value": {"before_pro": [True, False], "action_type": [1, 3]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 4
    }

    user_fav_count = {
        "entity": "user_id",
        "action_type": [1, 3],
        "name_to_save": "user_fav_count.csv",
        "interesting_value": {"before_pro": [True, False], "action_type": [1, 3]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3
    }

    user_seller_fav_count = {
        "entity": "user_seller",
        "action_type": [1, 3],
        "name_to_save": "user_seller_fav_count.csv",
        "interesting_value": {"before_pro": [True, False], "action_type": [1, 3]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3
    }

    user_buy_pro_count = {
        "entity": "user_id",
        "action_type": [2],
        "name_to_save": "user_buy_pro_count.csv",
        "interesting_value": {"before_pro": [True, False]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3
    }

    user_click_pro_count = {
        "entity": "user_id",
        "action_type": [0],
        "name_to_save": "user_click_pro_count.csv",
        "interesting_value": {"before_pro": [True, False]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3
    }

    seller_buy_pro_count = {
        "entity": "seller_id",
        "action_type": [2],
        "name_to_save": "seller_buy_pro_count.csv",
        "interesting_value": {"before_pro": [True, False]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3
    }

    seller_click_pro_count = {
        "entity": "seller_id",
        "action_type": [0],
        "name_to_save": "seller_click_pro_count.csv",
        "interesting_value": {"before_pro": [True, False]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3
    }

    user_seller_click_pro_count = {
        "entity": "user_seller",
        "action_type": [0],
        "name_to_save": "user_seller_click_pro_count.csv",
        "interesting_value": {"before_pro": [True, False]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id"]
    }

    user_seller_click_trend = {
        "entity": "user_seller",
        "action_type": [0],
        "name_to_save": "user_seller_click_trend.csv",
        "interesting_value": {"month": [6, 8, 11]},
        "agg_pre": ["count", "trend"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id"]
    }

    user_seller_fav_count_per_month = {
        "entity": "user_seller",
        "action_type": [3],
        "name_to_save": "user_seller_fav_count_per_month.csv",
        "interesting_value": {"month": [6, 8, 11]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id"]
    }

    user_seller_cart_count_per_month = {
        "entity": "user_seller",
        "action_type": [1],
        "name_to_save": "user_seller_cart_count_per_month.csv",
        "interesting_value": {"month": [6, 8, 11]},
        "agg_pre": ["count", "mean", "std"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id"]
    }

    user_seller_all_time_mean = {
        "entity": "user_seller",
        "action_type": [0, 1, 2, 3],
        "name_to_save": "user_seller_all_time_mean.csv",
        "interesting_value": {"month": [6, 8, 11]},
        "agg_pre": ["mean", "std"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id"],
        "trans_pre": [TimeSincePrevious]
    }

    user_seller_all_per_month_count_time = {
        "entity": "user_seller",
        "action_type": [0, 1, 2, 3],
        "name_to_save": "user_seller_all_per_month_count.csv",
        "interesting_value": {"month": [6, 9, 10, 11]},
        "agg_pre": [NUnique, "count", "mean", "std"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id", "time_stamp"],
    }

    user_seller_click_unique_count = {
        "entity": "user_seller",
        "action_type": [0],
        "name_to_save": "user_seller_click_unique_count.csv",
        "interesting_value": {"month": [6, 9, 10, 11]},
        "agg_pre": [NUnique, "mean", "std"],
        "depth": 3,
        "sub_entity_list": ["cat_id", "brand_id", "item_id"],
    }

    extraction(**user_seller_click_unique_count)

    for configuration in [user_seller_fav_count_per_month, user_seller_cart_count_per_month, user_seller_all_time_mean,
                          user_seller_all_per_month_count_time, user_seller_click_unique_count]:
        extraction(**configuration)

