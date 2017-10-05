import argparse
import numpy as np
import pandas as pd
import time
from popularity import PopularityBasedRecommendAlgorithm
from item_knn import ItemKnnRecommendAlgorithm
import requests

SERVER_URL = 'http://127.0.0.1:6789'


def run_model_for_test_data(model_name, k, test_numbers=1, use_remote=False):
    # prepare test data
    test_click_data = pd.read_csv('./data/yoochoose-test.dat', header=None,
                                  names=['Session_ID', 'Timestamp', 'Item_ID', 'Category'],
                                  dtype={'Session_ID': int, 'Timestamp': str, 'Item_ID': int,
                                         'Category': str})
    test_data_grouped = test_click_data.groupby('Session_ID')
    sessions = test_data_grouped.groups.keys()

    if not use_remote:
        model = models[model_name]
        model.load_model()

    # begin test
    hits = []            # number of recommended items that users actually purchased
    recommends = []      # number of recommended items
    purchased = []       # number of items that users purchased
    start = time.time()
    for session_idx, session in enumerate(sessions):
        if session_idx == test_numbers:
            break

        one_group = test_data_grouped.get_group(session)
        prev_half = (len(one_group) - 1) // 2 + 1
        if use_remote:
            recommendations = requests.post(SERVER_URL+'/model/{0}/test'.format(model_name), json={
                "session": int(session),
                "source_items": one_group[0:prev_half].to_json(),
                "k": k
            }).json()
        else:
            recommendations = model.recommend_items(session, one_group[0:prev_half], k)
        hit = 0
        for one_item in one_group['Item_ID'][prev_half:]:
            if one_item in recommendations:
                hit += 1
        # if hit/(len(one_group) - prev_half)>0.2 and (len(one_group) - prev_half)>2:
        #     pass
        hits.append(hit)
        recommends.append(k)
        purchased.append(len(one_group) - prev_half)
    return np.sum(hits)/np.sum(recommends), np.sum(hits)/np.sum(purchased), test_numbers/(time.time()-start)


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm',
                    type=str,
                    default='item_knn',
                    help='choose the recommendation algorithm you want to train')
parser.add_argument('-k',
                    type=int,
                    default=5,
                    help='choose the number of recommended items')
parser.add_argument('-n', '--num',
                    type=int,
                    default=1000,
                    help='choose the number of test sessions')

parser.add_argument('-s', '--use_server',
                    type=bool,
                    default=False,
                    help='choose recommend by server')

args = parser.parse_args()

models = {
    "popularity": PopularityBasedRecommendAlgorithm(),
    "item_knn": ItemKnnRecommendAlgorithm()
}


if args.algorithm in models:
    print('you choose to test {0} model...'.format(args.algorithm))
    precision, recall, throughput = run_model_for_test_data(
        model_name=args.algorithm,
        k=args.k,
        test_numbers=args.num,
        use_remote=args.use_server)
    print('model test finished, precision={0}, recall={1}, throughput={2}'.format(precision, recall, throughput))
else:
    raise ValueError('Not Implemented Algorithm')


