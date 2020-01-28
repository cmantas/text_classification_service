from os import listdir
from os.path import isfile, join, split
import pickle
from tensorflow.keras.models import load_model as load_keras_model

def load_model(prefix):
    obj = pickle.load(open(prefix + '.pickle', 'rb'))
    keras_model = load_keras_model(prefix + '.h5')
    obj.model = keras_model
    return obj

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
