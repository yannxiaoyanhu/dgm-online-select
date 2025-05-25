import numpy as np

from .pak_ucb import polynomial_kernel, rbf_kernel, update_inverse
from .rff_ucb import update_inverse_gram


class random_selector:
    def __init__(self, G, **kwargs):
        self.G = G

    def select_arm(self, **kwargs):
        return np.random.randint(self.G)

    def update_stats(self, **kwargs):
        pass

    def update_model_pool(self):
        self.G = self.G + 1


class greedy_selector:
    def __init__(self, G, **kwargs):
        self.G = G
        self.emp_score = + np.inf * np.ones((G,))
        self.visitation = np.zeros((G,), dtype=int)

    def select_arm(self, **kwargs):
        return np.random.choice(np.where(self.emp_score == np.max(self.emp_score))[0])

    def update_stats(self, g, reward, **kwargs):
        if self.visitation[g] == 0:
            self.emp_score[g] = reward
            self.visitation[g] = 1
        else:
            self.emp_score[g] = (self.emp_score[g] * self.visitation[g] + reward) / (self.visitation[g] + 1)
            self.visitation[g] += 1

    def update_model_pool(self):
        self.G = self.G + 1
        self.emp_score = np.concatenate((self.emp_score, np.array([+ np.inf], dtype=float)))
        self.visitation = np.concatenate((self.visitation, np.zeros((1,), dtype=int)))


class kernel_ucb:
    def __init__(self, G: int, num_dim: int, delta=.05,
                 kernel_method='poly', reg_alpha=1., kernel_para_c=1., kernel_para_d=3., kernel_para_gamma=1.,
                 exp_eta=None, **kwargs):
        self.G = G
        self.num_dim = num_dim
        self.krr_alpha = reg_alpha
        self.exp_eta = 2. * np.log(2. * G / delta) if exp_eta is None else exp_eta

        self.krr_inverse_mat = None
        self.krr_variable = None
        self.krr_target = np.empty((1,))
        self.visitation = np.zeros((G,), dtype=int)

        self.kernel_method = kernel_method
        if kernel_method == 'lin':
            self.c = 0.
            self.d = 1.
        else:
            self.c = kernel_para_c
            self.d = kernel_para_d
        self.gamma = kernel_para_gamma

    def kernel_function(self, x1, x2):
        if self.kernel_method in ['lin', 'poly']:
            return polynomial_kernel(x1=x1, x2=x2, c=self.c, d=self.d, gamma=self.gamma)
        elif self.kernel_method == 'rbf':
            return rbf_kernel(x1=x1, x2=x2, gamma=self.gamma)
        else:
            raise NotImplementedError

    def one_hot_rep(self, context, g):
        one_hot_g = np.zeros((self.G,))
        one_hot_g[g] = 1.
        return np.concatenate((context, one_hot_g.reshape(1, -1)), axis=1)

    def select_arm(self, context: np.array):
        assert context.ndim == 2 and context.shape[0] == 1

        if not np.all(self.visitation):
            return np.random.choice(np.where(self.visitation == 0)[0])

        ucb_values = np.empty((self.G,))
        for g in range(self.G):
            context_g = self.one_hot_rep(context=context, g=g)
            kernel_vector_g = np.empty((np.sum(self.visitation),))
            for n in range(np.sum(self.visitation)):
                kernel_vector_g[n] = self.kernel_function(x1=context_g[0], x2=self.krr_variable[n])

            mu_g = kernel_vector_g.T @ self.krr_inverse_mat @ self.krr_target
            sig_g = kernel_vector_g.T @ self.krr_inverse_mat @ kernel_vector_g
            sig_g = (self.krr_alpha ** -0.5) * np.sqrt(max(0., self.kernel_function(x1=context_g[0],
                                                                                    x2=context_g[0]) - sig_g))
            ucb_values[g] = mu_g + self.exp_eta * sig_g
        return np.random.choice((np.where(ucb_values == np.max(ucb_values)))[0])

    def update_stats(self, g: int, context: np.array, reward: float):
        assert context.ndim == 2 and context.shape[0] == 1
        context_g = self.one_hot_rep(context=context, g=g)

        if self.krr_inverse_mat is None:
            self.krr_inverse_mat = np.array(
                [1. / self.kernel_function(x1=context_g[0], x2=context_g[0]) + self.krr_alpha]).reshape((1, 1))
            self.krr_variable = context_g
            self.krr_target[0] = reward
        else:
            self.krr_inverse_mat = update_inverse(K_inv=self.krr_inverse_mat,
                                                  X=self.krr_variable,
                                                  x_new=context_g[0],
                                                  kernel=self.kernel_function,
                                                  alpha=self.krr_alpha)
            self.krr_variable = np.concatenate((self.krr_variable, context_g), axis=0)
            self.krr_target = np.concatenate((self.krr_target, np.array([reward])), axis=0)
        self.visitation[g] += 1


class lin_ucb:
    def __init__(self, G: int, num_dim: int, delta=.05, reg_alpha=1., exp_eta=None, **kwargs):
        self.G = G
        self.num_dim = num_dim
        self.lrr_alpha = reg_alpha
        self.exp_eta = 2. * np.log(2. * G / delta) if exp_eta is None else exp_eta

        self.lrr_inverse_mat = None
        self.b = None
        self.weight_vector = None
        self.visitation = np.zeros((G,), dtype=int)

    def one_hot_rep(self, context, g):
        one_hot_g = np.zeros((self.G,))
        one_hot_g[g] = 1.
        return np.concatenate((context, one_hot_g.reshape(1, -1)), axis=1)

    def select_arm(self, context: np.array):
        assert context.ndim == 2 and context.shape[0] == 1

        if not np.all(self.visitation):
            return np.random.choice(np.where(self.visitation == 0)[0])

        ucb_values = np.empty((self.G,))
        for g in range(self.G):
            context_g = self.one_hot_rep(context=context, g=g)
            mu_g = context_g.dot(self.weight_vector)
            sigma_g = np.sqrt(context_g @ self.lrr_inverse_mat @ context_g.T)
            ucb_values[g] = mu_g + self.exp_eta * sigma_g
        return np.random.choice((np.where(ucb_values == np.max(ucb_values)))[0])

    def update_stats(self, g: int, context: np.array, reward: float):
        assert context.ndim == 2 and context.shape[0] == 1

        context_g = self.one_hot_rep(context=context, g=g)
        if self.lrr_inverse_mat is None:
            self.lrr_inverse_mat = np.linalg.inv(np.outer(context_g, context_g) +
                                                 self.lrr_alpha * np.eye(context_g.shape[1]))
            self.b = reward * context_g
        else:
            self.lrr_inverse_mat = update_inverse_gram(G_inv=self.lrr_inverse_mat, x_new=context_g)
            self.b += reward * context_g
        self.weight_vector = self.lrr_inverse_mat @ self.b.T

        self.visitation[g] += 1
