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
    from scipy.signal import resample
    dfs = []
    N_MAX = 3600
    for name,var in kwargs.items():
        if type(var) is list:
            var = np.array(var)
        if isinstance(var, float) or isinstance(var, int):
            var = np.array([var])
        if type(var) is tuple and len(var) == 3:
            for i, x in enumerate(var):
                xnan = np.concatenate((x,np.array([np.nan]*x.shape[0]).reshape(-1,1)),axis = 1)
                dfs.append(pd.DataFrame(xnan.reshape(-1,1), columns = [name + '_surf_' + str(i)]))
        elif type(var) is np.ndarray:
            if var.ndim == 1:
                data = var if var.shape[0] < N_MAX else resample(var,N_MAX)
                dfs.append(pd.DataFrame(data=data, columns=[name]))
            elif var.ndim == 2:
                for i, c in enumerate(var.T):
                    data = c if c.shape[0] < N_MAX else resample(c, N_MAX)
                    dfs.append(pd.DataFrame(data=data, columns=[name + '_' + str(i)]))
            else:
                raise Exception("ndarray must be 1 or 2 dimensional")
        else:
            print('Ignored ' + name)
    df = pd.concat(dfs, axis=1)
    if df.shape[0] > N_MAX:
        print('Found large rows, Might conflict with tikz surf plot')
    df.to_csv(fout)