from item_knn import ItemKnnRecommendAlgorithm
from old.test_base import run_model_for_test_data

result = run_model_for_test_data(ItemKnnRecommendAlgorithm(), k=5, test_numbers=100000)
print(result) # 0.310967905652





