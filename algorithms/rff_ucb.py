import numpy as np


def update_inverse_gram(G_inv, x_new):
    x_new = x_new.reshape(-1, 1)
    v = G_inv @ x_new
    alpha = 1 + x_new.T @ v
    G_new_inv = G_inv - np.outer(v, v) / alpha
    return G_new_inv


class RFFKernel:
    def __init__(self, input_dim, num_features, sigma, version='D'):

        self.input_dim = input_dim
        self.version = version
        if version == 'D':
            self.num_features = num_features
        elif version == '2D':
            self.num_features = 2 * num_features
        else:
            raise NotImplementedError
        self.sigma = sigma
        self.W = np.random.normal(scale=1.0 / sigma, size=(num_features, input_dim))
        self.b = np.random.uniform(0, 2 * np.pi, num_features)

    def transform(self, X):
        if self.version == 'D':
            projection = np.dot(X, self.W.T) + self.b
            Z = np.sqrt(2.0 / self.num_features) * np.cos(projection)
        elif self.version == '2D':
            projection = np.dot(X, self.W.T)
            Z = np.hstack([
                np.cos(2 * np.pi * projection),
                np.sin(2 * np.pi * projection)
            ])
        else:
            raise NotImplementedError

        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        return Z


class rff_ucb:
    def __init__(self, G: int, T: int, num_dim: int, num_rff_dim: int, delta=.05, kernel_method='rbf',
                 kernel_para_gamma=1., reg_alpha=1., exp_eta=None, rff_version='D', **kwargs):
        assert kernel_method in ['rbf']

        self.G = G
        self.num_dim = num_dim
        self.exp_eta = np.sqrt(2. * np.log(2 * G / delta)) if exp_eta is None else exp_eta
        self.lrr_inverse_mat = [None for _ in range(G)]
        self.b = [None for _ in range(G)]
        self.weight_vector = [None for _ in range(G)]
        self.lrr_alpha = reg_alpha
        self.visitation = np.zeros((G,), dtype=int)
        self.rff_kernel = RFFKernel(input_dim=num_dim, num_features=num_rff_dim,
                                    sigma=kernel_para_gamma, version=rff_version)
        self.kernel_method = kernel_method
        self.gamma = kernel_para_gamma

    def select_arm(self, context: np.array):
        assert context.ndim == 2 and context.shape[0] == 1

        if not np.all(self.visitation):
            return np.random.choice(np.where(self.visitation == 0)[0])

        ucb_values = np.empty((self.G,))
        rff_context = self.rff_kernel.transform(X=context)
        for g in range(self.G):
            mu_g = rff_context.dot(self.weight_vector[g])
            sigma_g = np.sqrt(rff_context @ self.lrr_inverse_mat[g] @ rff_context.T)
            ucb_values[g] = mu_g + self.exp_eta * sigma_g
        return np.random.choice((np.where(ucb_values == np.max(ucb_values)))[0])

    def update_stats(self, g: int, context: np.array, reward: float):
        assert context.ndim == 2 and context.shape[0] == 1
        rff_feat = self.rff_kernel.transform(X=context)

        if self.visitation[g] == 0:
            self.lrr_inverse_mat[g] = np.linalg.inv(np.outer(rff_feat, rff_feat) +
                                                    self.lrr_alpha * np.eye(self.rff_kernel.num_features))
            self.b[g] = reward * rff_feat
        else:
            self.lrr_inverse_mat[g] = update_inverse_gram(G_inv=self.lrr_inverse_mat[g], x_new=rff_feat)
        self.b[g] += reward * rff_feat

        self.weight_vector[g] = self.lrr_inverse_mat[g] @ self.b[g].T

        self.visitation[g] += 1

    def update_model_pool(self):
        self.lrr_inverse_mat += [None]
        self.b += [None]
        self.weight_vector += [None]
        self.visitation = np.concatenate((self.visitation, np.array([0], dtype=int)))
