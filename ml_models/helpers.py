from random import shuffle

def read_data(fname, token='#'):
    with open(fname) as f:
        lines = f.readlines()
        rv = []
        for l in lines:
            l = l.strip()
            elems = l.split(token)
            if len(elems) != 2:
                continue
            pname, cid = elems
            rv.append((pname, cid))
        return rv

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

