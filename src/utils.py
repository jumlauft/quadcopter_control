def data_out_dir(note=''):
    import os
    from datetime import datetime
    dateTimeObj = datetime.now()
    datestr = dateTimeObj.strftime("%Y_%m_%d-%H_%M_%S")
    dirstr = '../results'  +  note  #datestr + '_' +
    if not os.path.exists(dirstr):
        os.mkdir(dirstr)
    return dirstr + '/'


def store_data(gl, fout):
    import shelve
    with shelve.open(fout, 'n') as data_file:
        for key in gl:
            try:
                data_file[key] = gl[key]
            except:
                pass
    data_file.close()


def classvar2file(class_to_store, fout):
    import inspect
    import json
    att = inspect.getmembers(class_to_store,
                             lambda a: not (inspect.isroutine(a)))
    pdict = dict([a for a in att if not (a[0].startswith('_') or
                                         a[0].endswith('_'))])
    with open(fout, 'w') as json_file:
        json.dump(pdict, json_file)


def data2csv(fout, **kwargs):
    import pandas as pd
    import numpy as np
    dfs = []
    for name in kwargs:
        try:
            if type(kwargs[name]) is list:
                kwargs[name] = np.array(kwargs[name])
            if kwargs[name].ndim == 1:
                dfs.append(pd.DataFrame(data = kwargs[name], columns = [name]))
            elif kwargs[name].ndim == 2:
                for i, col in enumerate(kwargs[name].T):
                    dfs.append(pd.DataFrame(data=col, columns=[name + '_' + str(i)]))
            else:
                raise Exception("must be 1 or 2 dimensional")
        except:
            print('Ignored ' + name)
    pd.concat(dfs,axis=1).to_csv(fout)