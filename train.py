import argparse
import numpy as np
import pandas as pd
import time
from popularity import PopularityBasedRecommendAlgorithm
from item_knn import ItemKnnRecommendAlgorithm


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm',
                    type=str,
                    default='popularity',
                    help='choose the recommendation algorithm you want to train')
args = parser.parse_args()


models = {
    "popularity": PopularityBasedRecommendAlgorithm(),
    "item_knn": ItemKnnRecommendAlgorithm()
}


if args.algorithm in models:
    print('you choose to build {0} model...'.format(args.algorithm))
    start = time.time()
    models[args.algorithm].learn(pd.read_csv('./data/yoochoose-clicks.dat', header=None,
                     names=['Session_ID', 'Timestamp', 'Item_ID', 'Category'],
                     dtype={'Session_ID': np.int64, 'Timestamp': str, 'Item_ID': np.int64, 'Category': str}))
    print('model built, training time is {0}'.format(time.time()-start))
else:
    raise ValueError('Not Implemented Algorithm')
