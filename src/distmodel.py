import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler


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


class DistModel:
    TRAIN_EPOCHS = 1
    TRAIN_ITER = 2
    N_HIDDEN = 50
    LEARNING_RATE = 0.01
    MOMENTUM = 0.0001
    SCALE_OFFSET = 1e-9
    RADIUS_TR = 0.0001
    MIN_ADD_DATA_RATE = 0.
    N_EPI = 1000
    TRAIN_LIM = 1

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
            x_epi (numpy array): input locations where no data is available
            y_epi (numpy array): output indicating high/low uncertainty
            _scaler (sklearn scaler): scaler for data
            loss (list): loss over training epochs
            _train_counter (int): counts number of added points until retraining

            TRAIN_EPOCHS (int): Number of training epochs per iteration
            TRAIN_ITER (int): Number of training iterations
            N_HIDDEN (int): Number of hidden neurons per layer
            LEARNING_RATE (float): step size of RMSprop optimizer
            MOMENTUM (float): momentum of RMSprop optimizer
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
        self.x_epi = self._generate_rand_epi(self.N_EPI)
        self.y_epi = np.ones((self.N_EPI, 1))
        self._scaler = StandardScaler()
        self._train_counter = 0
        self.loss = []

        self._scaler.fit(self.x_epi)
        self.model_out, self.model_epi, self.model_mean, self.model_ale, \
            self.model_mix, self.model_all = self._setup_nn

    def _generate_rand_epi(self, n):
        """ Generates random input locations for epistemic uncertainty measure

        Uniformly distributes data points across the input space defined by
        INPUT_LB and INPUT_UB


        Args:
            n (int): Number of points to be generated

        Returns:
            [n, DX] numpy array

        """
        lim = np.array([self.INPUT_LB, self.INPUT_UB])
        return (lim[1, :] - lim[0, :]) * np.random.rand(n, self.DX) + lim[0, :]

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
        epilay = Dense(1, activation='sigmoid')(hidden)

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
        model_epi = Model(inputs=inp, outputs=epilay)
        model_epi.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss='binary_crossentropy')
        model_epi.fit(self._scaler.transform(self.x_epi), self.y_epi,
                      epochs=self.TRAIN_EPOCHS, verbose=0)

        model_mix = Model(inputs=inp, outputs=[mulay, out])
        model_mix.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss=['mse', negloglik])
        model_mean = Model(inputs=inp, outputs=mulay)
        model_mean.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            loss='mse')
        model_ale = Model(inputs=inp, outputs=stdlay)

        model_all = Model(inputs=inp, outputs=[mulay, stdlay, epilay])
        return model_out, model_epi, model_mean, model_ale, model_mix, model_all

    def train(self):
        """ Trains the neural network based on the current data

        Training iterates between training the disturbance output and the
        epistemic uncertainty output

        """
        self._update_xy_epi()
        cw = compute_class_weight('balanced', np.unique(self.y_epi),
                                  self.y_epi.flatten())
        xepis = self._scaler.fit_transform(self.x_epi)
        xtrs = self._scaler.transform(self.Xtr)
        for i in range(self.TRAIN_ITER):
            # hist = self.model_out.fit(self.Xtr, self.Ytr, **kwargs)
            hist_epi = self.model_epi.fit(xepis, self.y_epi, class_weight=cw,
                                          epochs=self.TRAIN_EPOCHS, verbose=0)
            hist = self.model_mix.fit(xtrs, [self.Ytr, self.Ytr],
                                      epochs=self.TRAIN_EPOCHS, verbose=0)

            if np.isnan(hist_epi.history['loss']).any():
                print('detected Nan')
            self.loss = self.loss + hist.history['loss'] + hist_epi.history[
                'loss']

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
        mean, ale, epi = self.model_all.predict(self._scaler.transform(x))
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
        return self.model_mean.predict(self._scaler.transform(x)).flatten()

    @check_input
    def predict_aleatoric(self, x):
        """ Predicts aleatoric uncertainty of the NN model for the given input x

        Args:
            x: input

        Returns:
            aleatoric uncertainty prediction
        """
        return np.array(self.SCALE_OFFSET + tf.math.softplus(
            self.model_ale.predict(self._scaler.transform(x)))).flatten()

    @check_input
    def predict_epistemic(self, x):
        """ Predicts epistemic uncertainty of the NN model for the given input x

        Args:
            x: input

        Returns:
            epistemic uncertainty prediction
        """
        return self.model_epi.predict(self._scaler.transform(x)).flatten()

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

    def _update_xy_epi(self):
        """ Generates artificial data points for epistemic uncertainty estimate

        """
        ntr = self.Xtr.shape[0]

        # ALTERNATIVE 1
        # distance = np.sum((self.x_epi.reshape(1,-1,self.DX)
        #               - self.Xtr.reshape(ntr,1,self.DX))**2, axis=2)
        # dis1fill = distance.min(axis = 0).reshape(-1,1)
        # RADIUS_TR = 0.0001
        # self.y_epi = (dis1fill > RADIUS_TR).astype(int)

        # ALTERNATIVE 2
        # Generate uncertain points
        self.x_epi = self._generate_rand_epi(self.N_EPI + ntr)
        self.y_epi = np.ones((self.N_EPI + ntr, 1))

        # find closest uncertain points
        d = self.x_epi.reshape(1, -1, self.DX) - self.Xtr.reshape(-1, 1, self.DX)
        distance = np.sum(d ** 2, axis=2)
        idx = np.argpartition(distance.min(axis=0), ntr)

        # turn into certain points
        self.y_epi[idx[:ntr], :] = 0
        self.x_epi[idx[:ntr], :] = self.Xtr
