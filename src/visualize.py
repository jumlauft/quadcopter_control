import numpy as np
import matplotlib.pyplot as plt


def x_vs_xd(t, x, x_des):
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('x, x_des')
    plt.plot(t, x[:, 0:3])
    plt.plot(t, x_des, '--')
    plt.show()


def training_set(dmodel):
    data_fig = plt.figure()
    ax = data_fig.add_subplot(111, projection='3d')
    ax.set_title('Measured disturbance')
    ax.scatter(dmodel.Xtr[:, 0], dmodel.Xtr[:, 1], dmodel.Ytr)
    plt.show()


def dis_model_surf(dmodel):
    ndte = 50
    x1, x2 = np.meshgrid(np.linspace(-0.1, 0.2, ndte),
                         np.linspace(-0.1, 0.2, ndte))
    x = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)

    mean, ale, epi = dmodel.predict(x)

    modelfig = plt.figure(figsize=(15, 5))
    ax = modelfig.add_subplot(131, projection='3d')
    ax.set_title('Model Mean')
    ax.plot_surface(x1, x2, mean.reshape(ndte, ndte), alpha=0.5)

    ax = modelfig.add_subplot(132, projection='3d')
    ax.set_title('Model Aleatoric Uncertainty')
    ax.plot_surface(x1, x2, ale.reshape(ndte, ndte), alpha=0.5)

    ax = modelfig.add_subplot(133, projection='3d')
    ax.set_title('Model Epistemic Uncertainty')
    ax.plot_surface(x1, x2, epi.reshape(ndte, ndte), alpha=0.5)
    plt.show()


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
    plt.show()
    return mean_model, ale_model, epi_model, mean_true, ale_true


def training_loss(dmodel):
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(dmodel.loss)
    plt.show()
