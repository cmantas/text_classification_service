# load Flask
import flask
from flask import jsonify
from inspect import getmembers, isclass, isabstract

import ml_models


app = flask.Flask(__name__)

def available_models():
    models = []
    for _name, obj in getmembers(ml_models):
        if isclass(obj) and not isabstract(obj):
            models.append(obj)
    return models


model_classes = { m.__name__: m for m in available_models() }

@app.route('/model_types', methods=['GET'])
def model_types():
    rsp = list(model_classes.keys())
    # return a response in json format
    return jsonify(rsp)

# start the flask app, allow remote connections
app.run(host='0.0.0.0', debug=True)
