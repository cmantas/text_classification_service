def read_data(fname, delimiter="\t"):
    with open(fname) as f:
        lines = f.readlines()
        rv = []
        for l in lines:
            l = l.strip()
            elems = l.split(delimiter)
            rv.append(elems)
        return rv

def cast_data(data):
    rv = []
    for d in data:
        try:
            if len(d) == 3:
                rv.append([d[0].lower(), *map(int, d[1:])])
        except:
            pass
    return rv

def read_dataset(fname, delimiter="\t"):
    data = read_data(fname, delimiter=delimiter)
    data = cast_data(data)
    return data
