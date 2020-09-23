from src import distmodel
import numpy as np


def test_setup_nn():
    dx, dy = 2,1
    dmodel = distmodel.DistModel(dx,dy, [-1,-1],[1,1])


def test_train():
    dx, dy = 2,1
    ntr = 100
    dmodel = distmodel.DistModel(dx,dy,[-1,-1],[1,1])
    xtr = np.random.randn(ntr,dx)
    ytr = np.random.randn(ntr,dy)

    dmodel.add_data(xtr, ytr)
    dmodel.train()


def test_predict():
    dx, dy = 2,1
    nte = 100
    dmodel = distmodel.DistModel(dx,dy, [-1,-1],[1,1])
    xte = np.random.randn(nte,dx)

    mean, ale, epi = dmodel.predict(xte)

    assert (dmodel.predict_mean(xte) == mean).all()
    assert (dmodel.predict_epistemic(xte) == epi).all()
    assert (dmodel.predict_aleatoric(xte) == ale).all()
