# coding: utf-8

# In[1]:


from typing import List

import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from pandas import DataFrame
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from deep_learning.data_convert import *
from feature_selector import FeatureSelector


class Classifier:
    scaler = None

    @staticmethod
    def set_scaler(scaler):
        Classifier.scaler = scaler

    @staticmethod
    def get_scaler():
        return Classifier.scaler


def read_feature_file(file_name) -> pd.DataFrame:
    print("Reading:", file_name)
    file_path = os.path.join(get_root_path(), "feature_vectors", file_name)
    df = pd.read_csv(file_path, index_col=0)
    df.columns = df.columns.map(lambda x: str(x) + file_name)
    print("feature length:", len(df.columns.values))
    return df


def select_features_without_label(features: pd.DataFrame, missing_threshold=0.90,
                                  correlation_threshold=1) -> pd.DataFrame:
    print(missing_threshold)
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
    f_df.columns = f_df.columns.map(lambda x: str(x) + "_dic")
    return f_df


def feature_transform_pipeline(file_name_list: list, dic_list: list = None, df_list: list = None,
                               choose_to_downcast=True) -> pd.DataFrame:
    features_df_list: List[DataFrame] = read_all_features(file_name_list) + ([
                                                                                 select_features_without_label(
                                                                                     dict_to_df(i))
                                                                                 for i in
                                                                                 dic_list] if dic_list is not None else []) + (
                                            [select_features_without_label(i) for i in
                                             df_list] if df_list is not None else [])
    features_df = merge_all_features_on_index(features_df_list)
    if choose_to_downcast:
        return select_features_without_label(features_df, correlation_threshold=1).apply(pd.to_numeric, axis=1,
                                                                                         downcast='float')
    else:
        return select_features_without_label(features_df, correlation_threshold=1)


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


def join_logs(logs: pd.DataFrame, catalog_feature_file_list=None, merchant_feature_file_list=None,
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
        merchant_catalog_features = feature_transform_pipeline(merchant_catlog_feature_file_list, [], [])
        logs = logs.join(merchant_catalog_features, on="seller_catalog", how="left", rsuffix="_mc")
    if merchant_brand_feature_file_list is not None and len(merchant_brand_feature_file_list) > 0:
        merchant_brand_features = feature_transform_pipeline(merchant_brand_feature_file_list, [], [])
        logs = logs.join(merchant_brand_features, on="seller_brand", how="left", rsuffix="_md")

    return logs


def random_split_train_set(logs: pd.DataFrame, split_percent=0.85):
    msk = np.random.rand(len(all_pairs)) < split_percent
    train_pair = all_pairs[msk]
    valid_pair = all_pairs[~msk]

    train_log = logs[logs["user_seller"].isin(np.add(np.array(train_pair["user_id"].map(lambda x: str(x) + "_")),
                                                     np.array(train_pair["seller_id"].map(
                                                         lambda x: str(x)))))].drop_duplicates(
        subset=["user_id", "seller_id"], keep='first')

    valid_log = logs[logs["user_seller"].isin(np.add(np.array(valid_pair["user_id"].map(lambda x: str(x) + "_")),
                                                     np.array(valid_pair["seller_id"].map(
                                                         lambda x: str(x)))))].drop_duplicates(
        subset=["user_id", "seller_id"], keep='first')

    train_label = train_log["label"]
    train_log = train_log.drop(columns=drop_list + ["label"])
    valid_label = valid_log["label"]
    valid_log = valid_log.drop(columns=drop_list + ["label"])
    return train_log, train_label, valid_log, valid_label


def split_test_from_valid_set(valid_data, label, keep_percent=0.5):
    msk = np.random.rand(len(valid_data.index)) < keep_percent
    valid_data_left = valid_data[msk]
    valid_label = label[msk]
    test_data = valid_data[~msk]
    test_label = label[~msk]

    return valid_data_left, valid_label, test_data, test_label



def get_train_set(logs: pd.DataFrame, catalog_feature_file_list=None, merchant_feature_file_list=None,
                  brand_feature_file_list=None,
                  user_feature_file_list=None, merchant_user_feature_file_list=None,
                  merchant_catlog_feature_file_list=None, merchant_brand_feature_file_list=None, cat_f=None, mer_f=None,
                  brd_f=None, usr_f=None, brand_em=None, cat_em=None, seller_em=None, user_em=None):
    logs = join_logs(logs, catalog_feature_file_list, merchant_feature_file_list,
                     brand_feature_file_list,
                     user_feature_file_list, merchant_user_feature_file_list,
                     merchant_catlog_feature_file_list, merchant_brand_feature_file_list, cat_f, mer_f,
                     brd_f, usr_f, brand_em, cat_em, seller_em, user_em)

    return random_split_train_set(logs)


def choose_test_set(logs: pd.DataFrame):
    test_log = logs[logs["user_seller"].isin(np.add(np.array(test_pairs["user_id"].map(lambda x: str(x) + "_")),
                                                    np.array(test_pairs["seller_id"].map(
                                                        lambda x: str(x)))))].drop_duplicates(
        subset=["user_id", "seller_id"], keep='first')

    test_data = test_log.drop(columns=drop_list + ["label"])
    return test_data, test_log


def get_test_set(logs: pd.DataFrame, catalog_feature_file_list=None, merchant_feature_file_list=None,
                 brand_feature_file_list=None,
                 user_feature_file_list=None, merchant_user_feature_file_list=None,
                 merchant_catlog_feature_file_list=None, merchant_brand_feature_file_list=None, cat_f=None, mer_f=None,
                 brd_f=None, usr_f=None, brand_em=None, cat_em=None, seller_em=None, user_em=None):
    logs = join_logs(logs, catalog_feature_file_list, merchant_feature_file_list,
                     brand_feature_file_list,
                     user_feature_file_list, merchant_user_feature_file_list,
                     merchant_catlog_feature_file_list, merchant_brand_feature_file_list, cat_f, mer_f,
                     brd_f, usr_f, brand_em, cat_em, seller_em, user_em)
    return choose_test_set(logs)


def train(train_log, train_label, valid_log, valid_label, time_name, classifier_class=0):
    """

    :param train_log:
    :param train_label:
    :param valid_log:
    :param valid_label:
    :param time_name:
    :param classifier_class: 0:lgbm, 1:xgboost, 2:linear, 3:catboost
    :return:
    """
    if classifier_class == 0:
        return lgbm_train(train_log, train_label, valid_log, valid_label, time_name)
    elif classifier_class == 1:
        return xgboost_train(train_log, train_label, valid_log, valid_label, time_name)
    elif classifier_class == 2:
        return linear_train(train_log, train_label, valid_log, valid_label, time_name)
    elif classifier_class == 3:
        return cat_train(train_log, train_label, valid_log, valid_label, time_name)


def predict(bst, test_data, classifier_class=0):
    if classifier_class == 0:
        return lgbm_predict(bst, test_data)
    elif classifier_class == 1:
        return xgboost_predict(bst, test_data)
    elif classifier_class == 2:
        return linear_predict(bst, test_data)
    elif classifier_class == 3:
        return cat_predict(bst, test_data)


def cat_predict(bst, test_data):
    return bst.predict(Pool(test_data))


def lgbm_predict(bst, test_data):
    y_test = bst.predict(test_data, num_iteration=bst.best_iteration)
    return y_test


def xgboost_predict(bst, test_data):
    dtest = xgb.DMatrix(data=test_data)
    y_test = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    return y_test


def linear_predict(bst, test_data):
    normal_test = Classifier.get_scaler().transform(test_data.fillna(0).values)
    y_test = bst.predict(normal_test)
    return y_test


def linear_train(train_log, train_label, valid_log, valid_label, time_name):
    train_log = train_log.fillna(0)
    valid_log = valid_log.fillna(0)
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(1.0, 99.0), copy=True)
    scaler.fit(train_log.values)
    Classifier.set_scaler(scaler)

    normal_train = scaler.transform(train_log.values)
    normal_valid = scaler.transform(valid_log.values)

    classifier = linear_model.LogisticRegression(class_weight='balanced', solver="sag", max_iter=5000, verbose=1,
                                                 n_jobs=2)
    classifier.fit(normal_train, train_label)

    y_valid = classifier.predict(normal_valid)
    y_train = classifier.predict(normal_train)

    fpr, tpr, thresholds = metrics.roc_curve(train_label, y_train, pos_label=1)
    a = metrics.auc(fpr, tpr)
    print("train auc", a)

    fpr, tpr, thresholds = metrics.roc_curve(valid_label, y_valid, pos_label=1)
    a = metrics.auc(fpr, tpr)
    print(a)

    return classifier, a


def cat_train(train_log, train_label, valid_log, valid_label, time_name):
    train_data = Pool(train_log, train_label)
    valid_data = Pool(valid_log, valid_label)

    model = CatBoostClassifier(iterations=1000,
                               depth=8,
                               learning_rate=0.01,
                               custom_loss=['AUC'],
                               logging_level='Verbose',
                               # use_best_model=True,
                               class_weights=(0.05, 0.95))

    model.fit(train_data, eval_set=(valid_data))

    y_valid = model.predict(valid_data)
    fpr, tpr, thresholds = metrics.roc_curve(valid_label, y_valid, pos_label=1)
    a = metrics.auc(fpr, tpr)
    print("valid", a)

    y_valid = model.predict(train_data)
    fpr, tpr, thresholds = metrics.roc_curve(train_label, y_valid, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("train", auc)

    joblib.dump(model, os.path.join(get_root_path(), "model", time_name, 'cat' + str(a) + '.data'))

    return model, a


def xgboost_train(train_log, train_label, valid_log, valid_label, time_name):
    valid_log, valid_label, test_data, test_label = split_test_from_valid_set(valid_log, valid_label)
    dtrain = xgb.DMatrix(data=train_log, label=train_label)
    dvalid = xgb.DMatrix(data=valid_log, label=valid_label)
    dtest = xgb.DMatrix(data=test_data)
    #     param = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 6,
    #              'lambda': 1, 'tree_method': 'hist',
    #              'subsample': 0.9, 'colsample_bytree': 0.9, 'min_child_weight': 500, 'eta': 0.07,
    #              'silent': True, "gamma": 1}
    param = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric': 'auc', 'max_depth': 3,
             'lambda': 10,
             'subsample': 0.80, 'colsample_bytree': 0.75, 'min_child_weight': 1000, 'eta': 0.8,
             'silent': True,
             # 'gpu_id': 0, 'tree_method': 'gpu_hist', 'max_bin': 16, "predictor": 'gpu_predictor',
             "gamma": 10}
    eval_list = [(dtrain, 'train'), (dvalid, 'eval')]
    num_round = 3000

    bst = xgb.train(param, dtrain, num_round, eval_list, early_stopping_rounds=30)
    ntree_limit = bst.best_ntree_limit
    print("limit", ntree_limit)
    joblib.dump(bst, 'xgb_model.data')
    bst.__del__()

    bst = joblib.load('xgb_model.data')
    y_valid = bst.predict(dvalid, ntree_limit=ntree_limit)
    fpr, tpr, thresholds = metrics.roc_curve(valid_label, y_valid, pos_label=1)
    a = metrics.auc(fpr, tpr)

    y_test = bst.predict(dtest, ntree_limit=ntree_limit)
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_test, pos_label=1)
    auc_test = metrics.auc(fpr, tpr)
    print("test auc", auc_test)

    joblib.dump(bst, os.path.join(get_root_path(), "model", time_name, 'xgb_model' + str(a) + '.data'))
    print(a)

    return bst, (auc_test, a)


def lgbm_train(train_log, train_label, valid_log, valid_label, time_name):
    valid_log, valid_label, test_data, test_label = split_test_from_valid_set(valid_log, valid_label)
    dtrain = lgb.Dataset(data=train_log, label=train_label)
    dvalid = dtrain.create_valid(data=valid_log, label=valid_label)
    num_round = 3000
    param = {'objective': 'binary', 'lambda_l2': 0.5,
             'learning_rate': 0.02, 'max_bin': 63, 'min_data_in_leaf': 500,
             'num_leaves': 62, "max_depth": 6, 'reg_alpha': 1,
             'metrics': 'auc'}

    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=200,
                    verbose_eval=True)
    print("limit", bst.best_iteration)

    y_valid = bst.predict(valid_log, num_iteration=bst.best_iteration)

    fpr, tpr, thresholds = metrics.roc_curve(valid_label, y_valid, pos_label=1)
    a = metrics.auc(fpr, tpr)

    y_valid = bst.predict(train_log, num_iteration=bst.best_iteration)
    fpr, tpr, thresholds = metrics.roc_curve(train_label, y_valid, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    y_test = bst.predict(test_data, num_iteration=bst.best_iteration)
    fpr, tpr, thresholds = metrics.roc_curve(test_label, y_test, pos_label=1)
    auc_test = metrics.auc(fpr, tpr)
    print("test auc", auc_test)

    bst.save_model(os.path.join(get_root_path(), "model", time_name, 'xgb_model' + str(a) + '.data'),
                   num_iteration=bst.best_iteration)
    print("final:", a)
    print("final train:", auc)
    return bst, (auc_test, a)


def valid_auc(train_log, train_label, valid_log, valid_label, time_name, classifier_class=0):
    return train(train_log, train_label, valid_log, valid_label, time_name, classifier_class)


def select_top_features(train_data):
    fs = FeatureSelector(train_data[0], train_data[1])
    fs.identify_zero_importance(task='classification', eval_metric='auc',
                                n_iterations=6, early_stopping=True)
    fs.identify_low_importance(cumulative_importance=0.99)

    return fs.ops['zero_importance'], fs.ops['low_importance']


def data_normalization(logs: pd.DataFrame):
    kept_index = ['user_id', "seller_id", "user_seller", "label"]
    data = logs[logs.columns.difference(kept_index)]
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(0.1, 99.9), copy=True)
    values_normal = scaler.fit_transform(data.fillna(0).values)
    logs[logs.columns.difference(kept_index)] = values_normal
    return logs


def dimensionality_reduction(logs: pd.DataFrame) -> pd.DataFrame:
    kept_index = ['user_id', "seller_id", "user_seller", "label"]
    data = logs[logs.columns.difference(kept_index)]
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(0.1, 99.9), copy=True)
    values_normal = scaler.fit_transform(data.fillna(0).values)
    # data_normalization

    pca = PCA(n_components=int(len(logs.columns)*0.5), copy=False, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
    reduced_data = pca.fit_transform(values_normal)
    reduced_data_df = pd.DataFrame(reduced_data)
    return pd.concat([logs[kept_index], reduced_data_df], axis=1)


def drop_useless_data(logs: pd.DataFrame):
    logs.drop(["item_id", "cat_id", "brand_id", "seller_brand", "seller_catalog"], inplace=True, axis=1)


def tweak(logs, features, time_name, select_features=False, classifier_class=0, normalize_data=False, reduce_dimension=False):
    """

    :param logs:
    :param features:
    :param time_name:
    :param select_features:
    :param classifier_class:
    :param reduce_dimension:
    :return:
    """
    logs = join_logs(logs, **features)
    drop_useless_data(logs)
    print("final:", logs.shape)
    if select_features:
        print("Selecting features")
        low_importance = select_top_features(random_split_train_set(logs, 0.999))
        print(low_importance)
        print(len(low_importance[0] + low_importance[1]))
        logs.drop(labels=low_importance[0] + low_importance[1], inplace=True, axis=1)
    if normalize_data:
        logs = data_normalization(logs)
    if reduce_dimension:
        logs = dimensionality_reduction(logs)
    while True:
        train_data = random_split_train_set(logs)
        bst, auc = valid_auc(*train_data, time_name, classifier_class)
        test_set, test_log = choose_test_set(logs)
        y_test = predict(bst, test_set, classifier_class)
        test_log["prob"] = y_test
        test_log.groupby(["user_id", "seller_id"])["prob"].mean().reset_index().to_csv(
            os.path.join(get_root_path(), "prediction", time_name, "prob_x" + str(auc) + ".csv"), encoding='utf-8',
            index=False)


# def predict(test_log, *args):
#     dtest = xgb.DMatrix(data=test_log)
#     bst = joblib.load(os.path.join(get_root_path(), "model", *args))
#     y_test = bst.predict(dtest, bst.best_ntree_limit)
#     return y_test


def restore(features, time_name):
    test_data, test_log = get_test_set(all_log, **features)

    model_name_list = get_file_list_in_dir(os.path.join(get_root_path(), "model", time_name))
    for name in model_name_list:
        print(name)
        y_test = predict(test_data, time_name, name)
        test_log["prob"] = y_test
        test_log.groupby(["user_id", "seller_id"])["prob"].mean().reset_index().to_csv(
            os.path.join(get_root_path(), "prediction", time_name, "prob_x" + name + ".csv"), encoding='utf-8',
            index=False)


def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


if __name__ == '__main__':
    # from MF import get_buy_list
    # mf = get_buy_list()
    drop_list = ["seller_id", "user_id", "user_seller"]

    user_info_dict = get_user_info()

    # user_info_one_hot = pd.get_dummies(user_info_dict, columns=["age_range", "gender"])
    user_info_one_hot = user_info_dict

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
    all_log = pd.merge(all_log, user_info_one_hot, how="left", on="user_id")

    embedding_df_dict_usb = get_dhne_embedding("usb_128_5.npy", "train_dict_usb_s")

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

    # od = operate_days(-1).str.len().to_frame()
    # p_d = purchase_days(-1).str.len().to_frame()

    user_based_feature = load_dict_for_embedding("u_f_d.txt")
    # read of base log and feature has finished

    feature_list = dict()

    feature_list["merchant_feature_file_list"] = [
        "seller_ratio_list",
    ]
    #
    # feature_list["catalog_feature_file_list"] = [
    #     "catalogs_time.csv",
    #     "catlogs_buy_repeat_no.csv",
    #     "catlogs_buy_month.csv"
    #     #         "catalogs_buy_time_deep.csv"
    # ]
    #
    # feature_list["brand_feature_file_list"] = [
    #     "brands_time.csv",
    #     "brands_buy_repeat_no.csv",
    #     "brands_buy_month.csv"
    # ]
    #
    feature_list["user_feature_file_list"] = [
        "user_ratio_list"

                                              ]
    #
    feature_list["merchant_user_feature_file_list"] = [
        "user_seller_ratio"

    ]
    # feature_list["merchant_catlog_feature_file_list"] = [
    #     "seller_cat_buy_unique.csv",
    #     "seller_cat_buy_reapeat.csv",
    #     "seller_cat_buy_time.csv"
    # ]
    # #
    # feature_list["merchant_brand_feature_file_list"] = [
    #     "seller_brand_buy_unique.csv",  # good
    #     "seller_brand_buy_time.csv",
    #     "seller_brand_buy_repeat.csv"
    # ]
    #
    # feature_list["brand_em"] = [
    #     b_d,
    #     b_d_buy]
    # feature_list["cat_em"] = [
    #     c_d,
    #     c_d_buy
    # ]
    #
    # feature_list["seller_em"] = [
    #     m_d,
    #     m_d_buy
    # ]
    # feature_list["user_em"] = [
    #     user_based_feature
    # ]
    # feature_list["usr_f"] = [
    #     p_d,
    #     # embedding_df_dict_us["user_id"],
    #     # embedding_df_dict_usbc["user_id"],
    #     embedding_df_dict_usb["user_id"],
    #     od,
    # ]

    # feature_list["usr_f"] = [
    #     embedding_df_dict_usb["user_id"],
    # ]
    #
    # feature_list["mer_f"] = [
    #     embedding_df_dict_usb["seller_id"],
    # ]
    #
    # feature_list["brd_f"] = [
    #     embedding_df_dict_usb["brand_id"],
    # ]

    directory = "2.26"

    make_dir(os.path.join(get_root_path(), "model", directory))
    make_dir(os.path.join(get_root_path(), "prediction", directory))

    tweak(all_log, feature_list, directory, classifier_class=1, normalize_data=False, reduce_dimension=False)
    # restore(feature_list, directory)
    #
