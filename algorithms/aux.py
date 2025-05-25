from .pak_ucb import pak_ucb
from .rff_ucb import rff_ucb

from .baselines import random_selector, greedy_selector
from .baselines import lin_ucb, kernel_ucb


LEARNER = {
    "random": random_selector,
    "greedy": greedy_selector,

    "pak-ucb": pak_ucb,
    "rff-ucb": rff_ucb,
    "lin-ucb": lin_ucb,
    "kernel-ucb": kernel_ucb,
}
