from abc import ABC, abstractmethod


class BaseRecommendAlgorithm(ABC):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def learn(self, train_data):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def recommend_items(self, user_id, source_items, k):
        pass
