from old.test_base import run_model_for_test_data
from popularity import PopularityBasedRecommendAlgorithm

result = run_model_for_test_data(PopularityBasedRecommendAlgorithm(), k=10, test_numbers=10000)
print(result) # 0.004