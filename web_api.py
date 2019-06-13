# load Flask
import flask
from flask import jsonify
from inspect import getmembers, isclass, isabstract
from ml_models.helpers import load_models

import ml_models


app = flask.Flask(__name__)

SAVED_MODELS_DIRECTORY = 'data/saved_models'

def list_model_types():
    models = []
    for _name, obj in getmembers(ml_models):
        if isclass(obj) and not isabstract(obj):
            models.append(obj)
    return models


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

# start the flask app, allow remote connections
app.run(host='0.0.0.0', debug=True)
