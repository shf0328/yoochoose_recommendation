import numpy as np
import pandas as pd

from popularity import PopularityBasedRecommendAlgorithm

popularity_model = PopularityBasedRecommendAlgorithm()
click_data = pd.read_csv('./data/yoochoose-clicks.dat', header=None,
                     names=['Session_ID', 'Timestamp', 'Item_ID', 'Category'],
                     dtype={'Session_ID': np.int64, 'Timestamp': str, 'Item_ID': np.int64, 'Category': str})
popularity_model.learn(click_data)