from random import shuffle
from os import listdir
from os.path import isfile, join, split
import pickle
from inspect import getmembers, isclass, isabstract
import ml_models
import re


def read_data(fname, token='#'):
    with open(fname) as f:
        lines = f.readlines()
        rv = []
        for l in lines:
            l = l.strip()
            elems = l.split(token)
            rv.append(elems)
        return rv

def read_weighted_data(fname, token='#', limit=None):
    rv = []
    with open(fname) as f:
        lines = f.readlines()
        rv = []
        for l in lines:
            l = l.strip()
            elems = l.split(token)
            rv.append(elems)
    texts, labels, weights = zip(*rv)

    labels = [int(l) for l in labels]
    weights = [int(w) for w in weights]
    all_data = list(zip(texts, labels, weights))
    if limit is None:
        return all_data
    else:
        return all_data[:limit]

def binarize(data, balance=False, target_cid=40):
    positive, negative = [], []
    for line, cid in data:
        if cid == target_cid:
            positive.append((line, True))
        else:
            negative.append((line, False))
    shuffle(negative)
    if balance:
        negative = negative[:len(positive)]
    print("Returning %d positive and %d negative examples" %
      (len(positive), len(negative)))
    rv = negative + positive
    shuffle(rv)

    return rv


def list_model_types():
    models = []
    for _name, obj in getmembers(ml_models):
        if isclass(obj) and not isabstract(obj):
            models.append(obj)
    return models
