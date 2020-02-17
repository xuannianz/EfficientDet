from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np


class CosineAnnealingScheduler(Callback):
    def __init__(self, cycle_iterations, min_lr, t_mu=2, start_iteration=0):
        self.iteration_id = 0
        self.start_iteration = start_iteration
        self.cycle_iteration_id = 0
        self.lrs = []
        self.min_lr = min_lr
        self.cycle_iterations = cycle_iterations
        self.t_mu = t_mu
        super(CosineAnnealingScheduler, self).__init__()

    def on_batch_end(self, batch, logs):
        if self.iteration_id > self.start_iteration:
            # (1, 0)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (self.cycle_iteration_id / self.cycle_iterations)))
            decayed_lr = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr
            K.set_value(self.model.optimizer.lr, decayed_lr)
            if self.cycle_iteration_id == self.cycle_iterations:
                self.cycle_iteration_id = 0
                self.cycle_iterations = int(self.cycle_iterations * self.t_mu)
            else:
                self.cycle_iteration_id = self.cycle_iteration_id + 1
            self.lrs.append(decayed_lr)
        elif self.iteration_id == self.start_iteration:
            self.max_lr = K.get_value(self.model.optimizer.lr)
        self.iteration_id += 1

    def on_train_begin(self, logs={}):
        self.max_lr = K.get_value(self.model.optimizer.lr)


class ExponentialScheduler(Callback):
    def __init__(self, min_lr, max_lr, iterations):
        self.factor = np.exp(np.log(max_lr / min_lr) / iterations)
        self.min_lr = min_lr
        self.max_lr = max_lr
        # debug
        self.lrs = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, lr * self.factor)

    def on_train_begin(self, logs={}):
        K.set_value(self.model.optimizer.lr, self.min_lr)


class LinearWarmUpScheduler(Callback):
    def __init__(self, iterations, min_lr):
        self.iterations = iterations
        self.min_lr = min_lr
        self.iteration_id = 0
        # debug
        self.lrs = []

    def on_batch_begin(self, batch, logs):
        if self.iteration_id < self.iterations:
            lr = (self.max_lr - self.min_lr) / self.iterations * (self.iteration_id + 1) + self.min_lr
            K.set_value(self.model.optimizer.lr, lr)
        self.iteration_id += 1
        self.lrs.append(K.get_value(self.model.optimizer.lr))

    def on_train_begin(self, logs={}):
        self.max_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, self.min_lr)
        self.lrs.append(K.get_value(self.model.optimizer.lr))
