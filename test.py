import numpy as np
from termcolor import colored
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from algorithms.aux import LEARNER


G = 5
num_dim = 512
score_seq = np.random.random((G, 1000, 5))
best_model_on_avg = np.argmax(np.mean(np.mean(score_seq, axis=2), axis=1), axis=0)
contexts = np.random.randn(1000, num_dim)
max_score_seq = np.max(np.mean(score_seq, axis=2), axis=0)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--learner', type=str, default='rff-ucb')
parser.add_argument('--kernel_method', type=str, default='rbf')
parser.add_argument('--kernel_para_c', type=float, default=1.)
parser.add_argument('--kernel_para_d', type=float, default=3.)
parser.add_argument('--kernel_para_gamma', type=float, default=1.)
parser.add_argument('--reg_alpha', type=float, default=1.)
parser.add_argument('--num_rff_dim', type=int, default=200)
parser.add_argument('--rff_version', type=str, default='2D')
parser.add_argument('--exp_para', type=float, default=1.)
parser.add_argument('--eval_epochs', type=int, default=20)
parser.add_argument('--T', type=int, default=5000, help='Total number of generated samples')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')


def main():
    args = parser.parse_args()
    T = args.T
    num_epoch = args.eval_epochs
    np.random.seed(args.seed)

    # Track statistics
    o2b = np.zeros((T,))
    regret = np.zeros((T,))
    opr = np.zeros((T,))
    visitation = np.zeros((T, G,))

    for epoch in range(1, num_epoch + 1):

        learner = LEARNER[args.learner](G=G, T=T, num_dim=num_dim,
                                        kernel_method=args.kernel_method,
                                        kernel_para_c=args.kernel_para_c,
                                        kernel_para_d=args.kernel_para_d,
                                        kernel_para_gamma=args.kernel_para_gamma,
                                        num_rff_dim=args.num_rff_dim, rff_version=args.rff_version,
                                        reg_alpha=args.reg_alpha, exp_para=args.exp_para)

        for t in range(T):
            sample_idx_t = np.random.randint(contexts.shape[0])
            context_t = contexts[sample_idx_t].reshape(1, -1)
            best_model_on_avg_score_t = np.random.choice(score_seq[best_model_on_avg][sample_idx_t])
            max_score_t = max_score_seq[sample_idx_t]

            model_t = learner.select_arm(context=context_t)
            reward_t = np.random.choice(score_seq[model_t][sample_idx_t])
            learner.update_stats(g=model_t, context=context_t, reward=reward_t)

            o2b[t:] += reward_t - best_model_on_avg_score_t
            regret[t:] += max_score_t - reward_t
            visitation[t:, model_t] += 1
            opr[t:] += 1 if reward_t == max_score_seq[sample_idx_t] else 0

            if (t + 1) % 100 == 0:
                print(colored(f'Prompt-based Model Selection, '
                              f'learner: {args.learner}, kernel: {args.kernel_method}, '
                              f'epoch {epoch}, step {t + 1}', 'red'))
                print(colored(f'Outscore-the-best-model-on-avg: {o2b[t] / ((t + 1) * epoch)}, '
                              f'Regret-to-max-score: {regret[t] / ((t + 1) * epoch)}, '
                              f'OPR: {opr[t] / ((t + 1) * epoch)}, '
                              f'Visitation: {visitation[t] / ((t + 1) * epoch)}', 'blue'), '\n')


if __name__ == '__main__':
    main()
