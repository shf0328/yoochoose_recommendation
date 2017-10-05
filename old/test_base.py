import pandas as pd
from BaseRecommendAlgorithm import BaseRecommendAlgorithm
import numpy as np


def run_model_for_test_data(algorithm: BaseRecommendAlgorithm, k, test_numbers=None):
    algorithm.load_model()

    # prepare data
    test_click_data = pd.read_csv('./data/yoochoose-test.dat', header=None,
                                  names=['Session_ID', 'Timestamp', 'Item_ID', 'Category'],
                                  dtype={'Session_ID': np.int64, 'Timestamp': str, 'Item_ID': np.int64,
                                         'Category': str})
    test_data_grouped = test_click_data.groupby('Session_ID')
    sessions = test_data_grouped.groups.keys()

    # begin test
    i = 0  # count how many to test
    acc = []
    for session in sessions:
        i += 1
        if i == test_numbers:
            break

        one_group = test_data_grouped.get_group(session)
        prev_half = (len(one_group) - 1) // 2 + 1
        recommendations = algorithm.recommend_items(session, one_group[0:prev_half], k)

        hit = 0
        for one_item in one_group['Item_ID'][prev_half:]:
            if one_item in recommendations:
                hit += 1

        if len(one_group) - prev_half == 0:
            acc.append(0)
        else:
            acc.append(hit / (len(one_group) - prev_half))
    return np.mean(acc)