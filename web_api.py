# load Flask
import flask
from flask import jsonify, request
from ml_models.helpers import load_models, list_model_types

SAVED_MODELS_DIRECTORY = 'data/saved_models'

app = flask.Flask(__name__)


model_classes = {m.__name__: m for m in list_model_types()}

# Load saved models
saved_models = load_models(SAVED_MODELS_DIRECTORY)


@app.route('/model_types', methods=['GET'])
def model_types():
    rsp = list(model_classes.keys())
    # return a response in json format
    return jsonify(rsp)


@app.route('/saved_models', methods=['GET'])
def list_saved_models():
    rsp = {fname: model.describe() for fname, model in saved_models.items()}
    return jsonify(rsp)


@app.route('/model/<model_name>/predict', methods=['POST'])
def predict(model_name):
    params = request.json
    if model_name not in saved_models:
        return jsonify({ 'error': 'Model not found'})
    if 'texts' not in params:
        return jsonify({ 'error': 'No texts given to predict'})

    texts = params['texts']
    model = saved_models[model_name]
    probability = params.get('probabilities', False)
    predictions = model.predict(texts, probability)

    return jsonify(predictions)

# start the flask app, allow remote connections
app.run(host='0.0.0.0', debug=True)
