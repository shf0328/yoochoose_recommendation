from flask import Flask, request, jsonify
from item_knn import ItemKnnRecommendAlgorithm
from popularity import PopularityBasedRecommendAlgorithm
import pandas as pd

app = Flask(__name__)


models = {
    "popularity": PopularityBasedRecommendAlgorithm(),
    "item_knn": ItemKnnRecommendAlgorithm()
}


@app.route("/model/<model_name>/test", methods=['POST'])
def model_test(model_name):
    if model_name in models:
        recommendations = models[model_name].recommend_items(
            request.json['session'], pd.read_json(request.json['source_items']), request.json['k']
        )
        # fix int64 not serializable
        new_r = []
        for r in recommendations:
            new_r.append(int(r))
        return jsonify(new_r)
    else:
        return jsonify([])


if __name__ == '__main__':
    for name in models:
        models[name].load_model()

    app.run(host='127.0.0.1', port=6789)