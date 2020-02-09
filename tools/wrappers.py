import settings
from ml_models import FeedForwardMultiClassModel
from random import Random
from tools.data import *
from tools.models import *


MULTICLASS_TEXT_MODEL = FeedForwardMultiClassModel

def create_and_train_text_model(dataset_fname, model_name, epochs, limit=0,
                                seed=42, val_size=0, delim="\t"):
    dataset = read_dataset(dataset_fname, delim)
    print("Size:", len(dataset))

    if len(dataset) == 0:
        raise Exception(f"Empty dataset loaded from: \"{dataset_fname}\"")

    if limit != 0:
        dataset = dataset[:limit]

    Random(seed).shuffle(dataset)

    model = MULTICLASS_TEXT_MODEL.create_from_corpus(dataset)

    training_set = dataset

    model.train(training_set, val_size, epochs)

    model_fname = settings.saved_models_dir + '/' + model_name
    model.save(model_fname)

    return model.describe()


def saved_models():
    return load_models(settings.saved_models_dir)

def list_saved_models():
    all_models = saved_models()
    return describe_models(all_models)

def predict_with_model(model_name, dataset_fname, limit=0):
    models = saved_models()
    if model_name not in models:
        raise Exception(f"Model with name '{model_name}' not found")
    model = models[model_name]

    print(f"Using model '{model_name}'({type(model).__name__})")

    with open(dataset_fname, 'r') as f:
        texts = [l.strip() for l in f.readlines()]
        if limit != 0:
            texts = texts[:limit]

    predictions = model.predict_labels_with_probability(texts)
    return [[t, *p] for t,p in zip(texts, predictions)]
