from basic_model.feature_extraction.merchant_brand_features import repeat_buy_day_deep
from basic_model.feature_extraction.merchat_catalog_feature import repeat_buy_day_deep_mc
from basic_model.feature_extraction.auto_feature_extraction_user import UserFeatures
from basic_model.feature_extraction.category_profile import CatalogFeatures
from basic_model.classifiers.classfier_for_server_more_features import classifier
import logging

run_list = [repeat_buy_day_deep,
            repeat_buy_day_deep_mc,
            UserFeatures.extraction_click_repeat_deep,
            CatalogFeatures.buy_times_deep,
            classifier]

for func in run_list:
    try:
        print("runninng:", func.__name__)
        func()
    except:
        print("error:", func.__name__)
        logging.exception("message")
