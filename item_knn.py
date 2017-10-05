from BaseRecommendAlgorithm import BaseRecommendAlgorithm
import numpy as np
import scipy.sparse as sparse
import itertools
import pickle
import sklearn.preprocessing as preprocessing
import heapq

MODEL_SIMILARITY = './models/itemknn.similarity.npz'
MODEL_ID2IDX = './models/itemknn.item_id_to_idx.pickle'
MODEL_IDX2ID = './models/itemknn.idx_to_item_id.pickle'


class ItemKnnRecommendAlgorithm(BaseRecommendAlgorithm):
    def __init__(self):
        super().__init__('itemknn')
        self.similarity_mtx = None
        self.item_id_to_idx = None
        self.idx_to_item_id = None
        self.item_num = 0

    def learn(self, train_data):
        # map item id to idx [start from 0 to N]
        item_id_to_idx_map = {}
        i = 0
        for item in train_data['Item_ID']:
            if item not in item_id_to_idx_map:
                item_id_to_idx_map[item] = i
                i += 1
        # construct idx to item id map
        idx_to_item_id_map = {}
        for item_id in item_id_to_idx_map:
            idx_to_item_id_map[item_id_to_idx_map[item_id]]=item_id

        # Saving after construction:
        with open(MODEL_ID2IDX, 'wb') as f:
            pickle.dump(item_id_to_idx_map, f)

        with open(MODEL_IDX2ID, 'wb') as f:
            pickle.dump(idx_to_item_id_map, f)

        # build a sparse matrix
        mtx = sparse.dok_matrix((i, i), dtype=np.int32)

        # for every session, all items inside the session add similarity
        click_data_grouped = train_data.groupby('Session_ID')
        sessions = click_data_grouped.groups.keys()

        for num_session_processed, session in enumerate(sessions):
            if num_session_processed % 200000 == 0:
                print(num_session_processed)
            # get clicked items and add similarity
            one_group = click_data_grouped.get_group(session)
            ids = []
            for item_id in one_group['Item_ID']:
                ids.append(item_id_to_idx_map[item_id])
            for first, second in itertools.combinations(ids, 2):
                mtx[first, second] += 1
                mtx[second, first] += 1
        for item_idx in range(mtx.shape[0]):
            mtx[item_idx, item_idx] += 0
        mtx = preprocessing.normalize(mtx.tocsc(), norm='l1').todok()

        # after normalize, self similarity should be 1
        # for item_idx in range(mtx.shape[0]):
        #     mtx[item_idx, item_idx] += 1

        # save after construction
        sparse.save_npz(MODEL_SIMILARITY, mtx.tocoo())


    def load_model(self):
        self.similarity_mtx = sparse.load_npz(MODEL_SIMILARITY).tocsr()
        for item_idx in range(self.similarity_mtx.shape[0]):
            self.similarity_mtx[item_idx, item_idx] = 0
        self.item_num = self.similarity_mtx.shape[0]
        with open(MODEL_ID2IDX, 'rb') as f:
            self.item_id_to_idx = pickle.load(f)
        with open(MODEL_IDX2ID, 'rb') as f:
            self.idx_to_item_id = pickle.load(f)


    def recommend_items(self, user_id, source_items, k):
        d_u_i = sparse.dok_matrix((1, self.item_num), dtype=np.int8)
        for source_item in source_items['Item_ID']:
            if source_item in self.item_id_to_idx:
                d_u_i[0, self.item_id_to_idx[source_item]] = 1

        score = self.similarity_mtx.dot(d_u_i.tocsr().T).todok().items()
        recommendations = [self.idx_to_item_id[c[0][0]] for c in heapq.nlargest(k, score, key=lambda it: it[1])]
        return recommendations
