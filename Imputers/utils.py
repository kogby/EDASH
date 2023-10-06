import numpy as np
import pandas as pd
from functools import reduce

def simulate_nan(X, nan_rate):
    """(pd.dataframe, number) -> {str: pd.dataframe or number}

    Return the dictionary with four keys where:
    - Key 'X' stores a pd.dataframe where some of the entries in X
      are replaced with np.nan based on nan_rate specified.
    - Key 'C' stores a np.array where each entry is False if the
      corresponding entry in the key 'X''s np.array is np.nan, and True
      otherwise.
    """
    df = X
    X = X.to_numpy()
    # Create C matrix; entry is False if missing, and True if observed
    X_complete = X.copy()
    nr, nc = X_complete.shape
    C = np.random.random(nr * nc).reshape(nr, nc) > nan_rate

    # We don't want all components of a certain column is missing
    # Check for which i's we have all components become missing
    checker = np.where(sum(C.T) == 0)[0]
    if len(checker) == 0:
        # Every X_i has at least one component that is observed,
        # which is what we want
        X_complete[C == False] = np.nan
    else:
        # Otherwise, randomly bring back some components in such X_i's
        for index in checker:
            reviving_components = np.random.choice(
                nc, int(np.ceil(nc * np.random.random())), replace=False
            )
            C[index, np.ix_(reviving_components)] = True
        X_complete[C == False] = np.nan

    X_complete = pd.DataFrame(X_complete, columns=df.columns)

    result = {
        "X": X_complete,
        "C": C,
        "nan_rate": nan_rate,
        "nan_rate_actual": np.sum(C == False) / (nr * nc),
    }

    return result
