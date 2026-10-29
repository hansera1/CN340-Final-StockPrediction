import numpy as np
import pandas as pd


def qcol(q):
    return f"q{round(q * 100):02d}"


def as_frame(p_up, zq, quantiles, index):
    df = pd.DataFrame(np.sort(np.atleast_2d(zq), axis=1),   # sort = แก้ quantile crossing
                      columns=[qcol(q) for q in quantiles], index=index)
    df.insert(0, "p_up", np.clip(p_up, 1e-4, 1 - 1e-4))
    return df


def finite(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a, dtype=float))
    return mask
