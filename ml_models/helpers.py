from random import shuffle
from os import listdir
from os.path import isfile, join, split
import pickle

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

def vectorize_batch(batch, tokenizer, encoder):
    texts, cats = zip(*batch)
    X = tokenizer.texts_to_matrix(texts, mode='count')
    Y = encoder.transform(cats)
    return(X, Y)

def batcher(phrases, batch_size):
  for i in range(0, len(phrases), batch_size):
    frrom = i
    to = i+batch_size
    yield phrases[frrom:to]

def training_gen(texts, batch_size, tokenizer, label_encoder):
  while True:
    shuffle(texts)
    for batch in batcher(texts, batch_size):
      X, Y = vectorize_batch(batch, tokenizer, label_encoder)
      yield (X, Y)


def list_models(path):
    rv = []
    for f in listdir(path):
        fpath = join(path, f)
        if isfile(fpath) and f.endswith('.pickle'):
            rv.append(fpath)
    return rv


def load_models(models_dir):
    model_paths = list_models(models_dir)
    rv = {}
    for fpath in model_paths:
        fname = split(fpath)[-1]
        model = pickle.load(open(fpath, 'rb'))
        rv[fname] = model
    return rv
