import featuretools as ft
from data_munipulation.get_formatted_data import *
from featuretools.primitives import make_agg_primitive
from featuretools.variable_types import Categorical, Numeric


def count_repeat(column):
    from itertools import groupby
    full = set()
    sec = set()
    repeat = 0
    for i in column:
        if i in full:
            if i not in sec:
                repeat += 1
                sec.add(i)
        full.add(i)
    return repeat / len(set(column))


def repeat_percent(column):
    a = set(column)
    return len(a) / len(column)


def count_set_length(column):
    a = set(column)
    return len(a)


cunt_rpt = make_agg_primitive(function=count_repeat, input_types=[Categorical], return_type=Numeric)
CountDay = make_agg_primitive(function=count_set_length, input_types=[ft.variable_types.Datetime], return_type=Numeric)
RepeatPercent = make_agg_primitive(function=repeat_percent, input_types=[Categorical], return_type=Numeric)

log_df = get_train_log(None)
log_df = log_df.loc[log_df['action_type'] == 2]
log_df["user_seller"] = np.add(np.array(log_df["user_id"].map(lambda x: str(x) + "_")),
                               np.array(log_df["seller_id"].map(lambda x: str(x))))
log_df['data'] = log_df["time_stamp"].map(lambda x: '2016-' + str(int(x / 100)) + '-' + str(int(x // 100)))
log_df["month"] = log_df["time_stamp"].map(lambda x: int(x / 100))
user_df = get_user_info()
log_df = log_df.merge(user_df, on="user_id", how="inner")
log_df.drop(labels=['user_id', 'seller_id', 'action_type', 'age_range', 'gender'], axis=1, inplace=True)
log_df["index"] = log_df.index
es = ft.EntitySet(id="logs")
es = es.entity_from_dataframe(entity_id="logs",
                              dataframe=log_df,
                              index="index",
                              time_index="data",
                              variable_types={
                                  "item_id": ft.variable_types.Categorical,
                                  "cat_id": ft.variable_types.Categorical,
                                  "brand_id": ft.variable_types.Categorical,
                                  "month": ft.variable_types.Categorical,
                                  "time_stamp": ft.variable_types.Datetime,
                                  "user_seller": ft.variable_types.Categorical,
                              }
                              )

es = es.normalize_entity(base_entity_id="logs",
                         new_entity_id="user_seller",
                         index="user_seller")

es = es.normalize_entity(base_entity_id="logs",
                         new_entity_id="catalogs",
                         index="cat_id")

es = es.normalize_entity(base_entity_id="logs",
                         new_entity_id="time",
                         index="time_stamp")

es = es.normalize_entity(base_entity_id="logs",
                         new_entity_id="months",
                         index="month")

es = es.normalize_entity(base_entity_id="logs",
                         new_entity_id="brands",
                         index="brand_id")



feature_defs = ft.dfs(entityset=es,
                      target_entity="user_seller",
                      agg_primitives=["count", "mean", "max", "min", "std", "median"],
                      max_depth=3,
                      where_primitives=["count", "mean", "std"],
                      trans_primitives=[],
                      features_only=True
                      )
print(feature_defs)

print("start")
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="user_seller",
                                      agg_primitives=["count", "mean", "max", "min", "std", "median"],
                                      max_depth=3,
                                      where_primitives=["count", "mean", "std"],
                                      trans_primitives=[]
                                      # features_only=True
                                      )

print(feature_defs)
feature_matrix.to_csv("./seller_user_buy_daily.csv", float_format='%.4f', index_label="index")
