import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def check_input(func):
    """ Decorator to check the inputs

    Decorator which tests the input to the prediction functions for the
    correct dimensions.

    Args:
        func (callable): Function for which the input is checked
    """
    import functools

    @functools.wraps(func)
    def wrapper_check_input(self, x):
        # Convert input if possible
        if x.ndim == 1:
            x = x.reshape(-1, self.DX)
        n = x.shape[0]
        # Check for correct dimension
        assert x.shape == (n, self.DX)
        return func(self, x)

    return wrapper_check_input


class DistModel_NoEpi:
    TRAIN_EPOCHS = 2
    TRAIN_ITER = 2
    N_HIDDEN = 50
    LEARNING_RATE = 0.01
    SCALE_OFFSET = 1e-9
    MIN_ADD_DATA_RATE = 0.
    N_EPI = 1
    TRAIN_LIM = 1
    EPI_CONST = 0.145
    def __init__(self, dx, dy, input_lb, input_up):
        """ Online disturbance model to differentiate types of uncertainties

        Args:
            dx (int): input dimension
            dy (int): output dimension
            input_lb (list): length of DX list of minimal inputs
            input_up (list): length of DX list of maximal outputs

        Attributes:
            DX (int): input dimension
            DY (int): output dimension
            INPUT_LB (list): length of DX list of minimal inputs
            INPUT_UB (list): length of DX list of maximal outputs
            loss (list): loss over training epochs
            _train_counter (int): counts number of added points until retraining

            TRAIN_EPOCHS (int): Number of training epochs per iteration
            TRAIN_ITER (int): Number of training iterations
            N_HIDDEN (int): Number of hidden neurons per layer
            LEARNING_RATE (float): step size of RMSprop optimizer
            SCALE_OFFSET (float): numerical stability for variance predictions
            MIN_ADD_DATA_RATE (float): lower bounds the acceptance probability for
                                        incoming data points
            N_EPI (int): number of additional data points stored for epsistemic
            TRAIN_LIM (int): upper bound for _train_counter (triggers retraining)


        """
        self.DX = dx
        self.DY = dy
        self.INPUT_LB = input_lb
        self.INPUT_UB = input_up
        self._train_counter = 0
        self.loss = []
        self.x_epi = np.zeros((self.N_EPI,dx))
        self.y_epi = np.ones((self.N_EPI, 1))
        self.model_out, self.model_mean, self.model_ale, \
            self.model_mix, self.model_all = self._setup_nn

    @property
    def _setup_nn(self):
        """ Sets up the neural network structure for the disturbance model

        The neural network has three outputs
        - disturbance estimation
        - epsistemic uncertainty estimate
        - aleatoric uncertainty estimate

        Returns:
            model_out: model to output
            model_epi: model to epistemic uncertainty prediction
            model_mean: model to mean output prediction
            model_ale: model to aleatoric uncertainty prediction
            model_mix: model to mean and output
            model_all: model to all outputs at once

        """
        inp = Input(shape=(self.DX,))
        hidden = Dense(self.N_HIDDEN, activation="relu")(inp)
        hidden = Dense(self.N_HIDDEN, activation="relu")(hidden)

        mulay = Dense(self.DY)(hidden)
        stdlay = Dense(self.DY)(hidden)

        def negloglik(y, p_y):
            return -p_y.log_prob(y)

        def disturbance_distribution(p):
            return tfp.distributions.Normal(loc=p[..., :1],
                                            scale=tf.math.softplus(p[..., 1:]))

        dist = tfp.layers.DistributionLambda(disturbance_distribution)
        out = dist(tf.concat([mulay, stdlay], axis=1))

        model_out = Model(inputs=inp, outputs=out)
        model_out.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss=negloglik)

        model_mix = Model(inputs=inp, outputs=[mulay, out])
        model_mix.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss=['mse', negloglik])
        model_mean = Model(inputs=inp, outputs=mulay)
        model_mean.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss='mse')
        model_ale = Model(inputs=inp, outputs=stdlay)

        model_all = Model(inputs=inp, outputs=[mulay, stdlay])
        return model_out, model_mean, model_ale, model_mix, model_all

    def train(self):
        """ Trains the neural network based on the current data

        Training iterates between training the disturbance output and the
        epistemic uncertainty output

        """
        for i in range(self.TRAIN_ITER):
            # hist = self.model_out.fit(self.Xtr, self.Ytr, **kwargs)
            hist = self.model_mix.fit(self.Xtr, [self.Ytr, self.Ytr],
                                      epochs=self.TRAIN_EPOCHS, verbose=0)

            self.loss = self.loss + hist.history['loss']
        print('retrained disturbance model with ' + str(
            self.Xtr.shape[0]) + ' data points')

    @check_input
    def predict(self, x):
        """ Predicts all three outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean, aleatoric uncertainty, epistemic uncertainty
        """
        mean, ale = self.model_all.predict(x)
        epi = self.EPI_CONST*np.ones_like(ale)
        return mean.flatten(), self.SCALE_OFFSET + np.array(
            tf.math.softplus(ale.flatten())), epi.flatten()

    @check_input
    def predict_mean(self, x):
        """ Predicts mean outputs of the NN model for the given input x

        Args:
            x: input

        Returns:
            mean prediction
        """
        return self.model_mean.predict(x).flatten()

    @check_input
    def predict_aleatoric(self, x):
        """ Predicts aleatoric uncertainty of the NN model for the given input x

        Args:
            x: input

        Returns:
            aleatoric uncertainty prediction
        """
        return np.array(self.SCALE_OFFSET + tf.math.softplus(
            self.model_ale.predict(x))).flatten()

    @check_input
    def predict_epistemic(self, x):
        """ Predicts epistemic uncertainty of the NN model for the given input x

        Args:
            x: input

        Returns:
            epistemic uncertainty prediction
        """
        return self.EPI_CONST*np.ones(x.shape[0])

    def _select_data(self, x, epi_pred=None):
        """ Filters data added to the model based on epistemic uncertainty

        Args:
            x: input of data to be added
            (epi_pred): epistemic uncertainty prediction at x

        Returns:
            idx: index list (1 = store point, 0 = discard point)
        """
        if epi_pred is None:
            p = self.predict_epistemic(x).reshape(-1)
        else:
            p = epi_pred
        idx = np.random.binomial(1, np.clip(p + self.MIN_ADD_DATA_RATE, 0,
                                            1)) == 1
        return idx

    def add_data(self, xtr, ytr, epi_pred=None):
        """ Adds new training data points to the disturbance model

        Selects data to be added and triggers retraining if necessary

        Args:
            xtr: input of data to be added
            ytr: output of data to be added
            epi_pred: epistemic uncertainty prediction at xtr
        """
        if xtr.ndim == 1:
            xtr = xtr.reshape(-1, self.DX)
        if ytr.ndim == 1:
            ytr = ytr.reshape(-1, self.DY)
        n = xtr.shape[0]
        assert xtr.shape == (n, self.DX)
        assert ytr.shape == (n, self.DY)
        assert not np.isnan(xtr).any()
        assert not np.isnan(ytr).any()

        if not hasattr(self, 'Xtr'):
            # self.update_xy_epi(xtr)
            self.Xtr = xtr
            self.Ytr = ytr
            self._train_counter += n
        else:
            ii = self._select_data(xtr, epi_pred)
            # self.update_xy_epi(xtr[ii, :])
            self.Xtr = np.concatenate((self.Xtr, xtr[ii, :]), axis=0)
            self.Ytr = np.concatenate((self.Ytr, ytr[ii, :]), axis=0)
            self._train_counter += ii.sum()

        if self._train_counter >= self.TRAIN_LIM:
            self.train()
            self._train_counter -= self.TRAIN_LIM


    def evaluate(self, dis):
        ndtr = 20
        c1 = (0,0)
        c2 = (0.1,0.1)
        x1 = np.concatenate((np.linspace(c1[0], c2[0], ndtr), c2[0] * np.ones(ndtr),
                               np.linspace(c2[0], c1[0], ndtr), c1[0] * np.ones(ndtr)),
                              axis=0)
        x2 = np.concatenate((c1[1] * np.ones(ndtr), np.linspace(c1[1], c2[1], ndtr),
                             c2[1] * np.ones(ndtr), np.linspace(c2[1], c1[1], ndtr)),
                              axis=0)
        x = np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
        mean_model, ale_model, _ = self.predict(x)
        n_sample = 1000
        xt = np.tile(x, [1, 1, n_sample]).reshape(-1, 2)
        w = dis(xt)[:, 2].reshape(-1, n_sample)
        mean_true = w.mean(axis=1)
        ale_true = w.std(axis=1)
        return mean_squared_error(mean_model,mean_true),\
               mean_squared_error(ale_model, ale_true)