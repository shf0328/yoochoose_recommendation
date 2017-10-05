from BaseRecommendAlgorithm import BaseRecommendAlgorithm
import pickle


class PopularityBasedRecommendAlgorithm(BaseRecommendAlgorithm):
    def __init__(self):
        super().__init__('PopularityBased')
        self.popularity_map = None

    def learn(self, train_data):
        # map item id to idx [start from 0 to N]
        grouped = train_data.groupby('Category')
        categories = grouped.groups.keys()
        count_res = grouped['Item_ID'].value_counts()
        popularity_map = {}

        # select top 10 popular items for each model
        for c in categories:
            popularity_map[c] = count_res.loc[c][0:10].index.values.copy()

        # Saving the objects:
        with open('./models/popularity.category_popularity.pickle', 'wb') as f:
            pickle.dump(popularity_map, f)

    def load_model(self):
        with open('./models/popularity.category_popularity.pickle', 'rb') as f:
            self.popularity_map = pickle.load(f)

    def recommend_items(self, user_id, source_items, k):
        current_category = source_items['Category'].value_counts().index[0]
        if current_category in self.popularity_map:
            return set(self.popularity_map[current_category][0:k])
        else:
            return set([])
