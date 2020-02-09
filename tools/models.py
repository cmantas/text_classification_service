from os import listdir
from os.path import isfile, join, split
import pickle
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model as load_keras_model

def list_models(path):
    rv = []
    for f in listdir(path):
        fpath = join(path, f)
        if isfile(fpath) and f.endswith('.pickle'):
            rv.append(fpath[0:-7])
    return rv

def load_models(models_dir):
    model_paths = list_models(models_dir)
    rv = {}
    for fpath in model_paths:
        fname = split(fpath)[-1]
        model = load_model(fpath)
        rv[fname] = model
    return rv

def describe_models(models):
    return {fname: model.describe() for fname, model in models.items()}


def save_model(model, filename):
    keras_model = model.model
    keras_model_data = serialize_keras_model(keras_model)
    model.model = None
    intermediate = (model, keras_model_data)
    pickle.dump(intermediate, open(filename, 'wb'))
    model.model = keras_model # be non-destructive

def load_model(filename):
    intermediate = pickle.load(open(filename, 'rb'))
    model, keras_model_data = intermediate
    keras_model = deserialize_keras_model(keras_model_data)
    model.model = keras_model
    return model


def deserialize_keras_model(keras_model_data):
    """
    Deserializes a keras model from raw bytes
    """
    with NamedTemporaryFile(suffix='.h5') as tmp:
        tmp.write(keras_model_data)
        tmp.seek(0)
        return load_keras_model(tmp.name)


def serialize_keras_model(km):
    """
    Serializes a keras model to raw bytes
    """
    with NamedTemporaryFile(suffix='.h5') as tmp:
        km.save(tmp.name)
        tmp.seek(0)
        data = tmp.read()
        return data
