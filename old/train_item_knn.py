import numpy as np
import pandas as pd

from item_knn import ItemKnnRecommendAlgorithm

itemKnn = ItemKnnRecommendAlgorithm()
click_data = pd.read_csv('./data/yoochoose-clicks.dat', header=None,
                     names=['Session_ID', 'Timestamp', 'Item_ID','Category'],
                     dtype={'Session_ID': np.int64, 'Timestamp': str, 'Item_ID': np.int64,'Category': str})
itemKnn.learn(click_data)