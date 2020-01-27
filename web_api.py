# load Flask
import flask
from flask import jsonify, request
from ml_models.helpers import load_models, list_model_types, describe_models
from os.path import join as join_path

SAVED_MODELS_DIRECTORY = 'data/saved_models'
DATASET_FOLDER = 'data'

app = flask.Flask(__name__)

app.config['UPLOAD_FOLDER'] = DATASET_FOLDER

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
    rsp = describe_models(saved_models)
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


@app.route('/model/<model_name>/train', methods=['POST'])
def upload_file(model_name):
    # check if the post request has the file part
    if 'file' in request.files:
        file = request.files['file']
        filename = model_name + '_training_data'
        file.save(join_path(app.config['UPLOAD_FOLDER'], filename))

        return jsonify({'saved': 'OK'})

# start the flask app, allow remote connections
app.run(host='0.0.0.0', debug=True)
