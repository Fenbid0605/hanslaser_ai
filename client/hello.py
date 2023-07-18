import torch
from flask import Flask, request, jsonify
from ai.genetic import GA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
ga = GA()


@app.route('/api/predict', methods=['POST'])
def post_json():
    data = request.get_json()
    predict = ga.predict(torch.Tensor([data['l'], data['a'], data['b']]))
    return jsonify(predict), 200


if __name__ == '__main__':
    app.run(debug=True)
