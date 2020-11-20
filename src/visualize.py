import numpy as np
import matplotlib.pyplot as plt


def x_vs_xd(t, x, x_des):
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('x, x_des')
    plt.plot(t, x[:, 0:3])
    plt.plot(t, x_des, '--')
    rmse_z = np.sqrt(np.mean((x[:, 2]-x_des[:,2])**2))
    plt.title('RMSE_Z: ' + str(rmse_z))
    return rmse_z


def training_set(dmodel):
    data_fig = plt.figure()
    ax = data_fig.add_subplot(111, projection='3d')
    ax.set_title('Measured disturbance')
    ax.scatter(dmodel.Xtr[:, 0], dmodel.Xtr[:, 1], dmodel.Ytr)


def dis_model_surf(dmodel):
    ndte = 60
    x1, x2 = np.meshgrid(np.linspace(-0.1, 0.2, ndte),
                         np.linspace(-0.1, 0.2, ndte))
    xte = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)

    mean, ale, epi = dmodel.predict(xte)
    meanq, aleq, epiq = mean.reshape(ndte, ndte),ale.reshape(ndte, ndte),epi.reshape(ndte, ndte)
    modelfig = plt.figure(figsize=(15, 5))
    ax = modelfig.add_subplot(131, projection='3d')
    ax.set_title('Model Mean')
    ax.plot_surface(x1, x2, meanq, alpha=0.5)

    ax = modelfig.add_subplot(132, projection='3d')
    ax.set_title('Model Aleatoric Uncertainty')
    ax.plot_surface(x1, x2, aleq, alpha=0.5)

    ax = modelfig.add_subplot(133, projection='3d')
    ax.set_title('Model Epistemic Uncertainty')
    ax.plot_surface(x1, x2, epiq, alpha=0.5)
    # import tikzplotlib
    # tikzplotlib.save("test.tex")
    return xte, mean, ale, epi
    # return (x1,x2,mean),(x1,x2,ale),(x1,x2,epi)

def dis_model_x(dmodel, dis, x):
    mean_model, ale_model, epi_model = dmodel.predict(x)
    n_sample = 1000
    xt = np.tile(x, [1, 1, n_sample]).reshape(-1, 2)
    w = dis(xt)[:, 2].reshape(-1, n_sample)
    mean_true = w.mean(axis=1)
    ale_true = w.std(axis=1)

    modelfig = plt.figure(figsize=(15, 5))
    ax = modelfig.add_subplot(131, projection='3d')
    ax.set_title('Disturbance')
    hm = ax.scatter(x[:, 0], x[:, 1], mean_model, color="blue")
    ht = ax.scatter(x[:, 0], x[:, 1], mean_true, color="red")
    ax.legend((ht, hm), ('true', 'model'))

    ax = modelfig.add_subplot(132, projection='3d')
    ax.set_title('Aleatoric Uncertainty')
    ax.scatter(x[:, 0], x[:, 1], ale_model, color="blue")
    ax.scatter(x[:, 0], x[:, 1], ale_true, color="red")

    ax = modelfig.add_subplot(133, projection='3d')
    ax.set_title('Epistemic Uncertainty')
    ax.scatter(x[:, 0], x[:, 1], epi_model, color="blue")
    ax.scatter(dmodel.x_epi[:, 0], dmodel.x_epi[:, 1], dmodel.y_epi[:, 0],
               color='red')
    return mean_model, ale_model, epi_model, mean_true, ale_true


def training_loss(dmodel):
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(dmodel.loss)
def show():
    plt.show()

