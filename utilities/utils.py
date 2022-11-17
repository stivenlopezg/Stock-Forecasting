import numpy as np
import pandas as pd
from itertools import product


def bootstrap_distribution(x: pd.Series, statistic: str = "mean", n_iter: int = 1000):
    n = len(x)
    dist_boot = []
    for i in range(n_iter):
        resample = np.random.choice(x, size=n, replace=True)
        if statistic == 'mean':
            dist_boot.append(np.mean(a=resample))
        elif statistic == 'p50':
            dist_boot.append(np.median(a=resample))
        elif statistic == 'p25':
            dist_boot.append(np.quantile(a=resample, q=0.25))
        else:
            dist_boot.append(np.quantile(a=resample, q=0.75))
    return dist_boot


def combinator(items, r: int = 1):
    cmb = [i for i in product(*items, repeat=r)]
    return cmb
