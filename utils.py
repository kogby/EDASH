import numpy as np
import pandas as pd
from functools import reduce


def impute_em(X, max_iter=3000, eps=1e-05):
    # iid multivariate normal samples
    """(pd.dataFrame, int, number) -> {str: pd.dataFrame or int}

    Return the dictionary with five keys where:
    - Key 'mu' stores the mean estimate of the imputed data.
    - Key 'Sigma' stores the variance estimate of the imputed data.
    - Key 'X_imputed' stores the imputed data that is mutated from X using
      the EM algorithm.
    - Key 'C' stores the np.array that specifies the original missing entries
      of X.
    """

    cols = X.columns
    X = X.to_numpy()
    nr, nc = X.shape
    # num of rows, num of columns
    C = np.isnan(X) == False

    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step=1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1

    # Generate Mu_0 and Sigma_0
    # Obeserved mean only
    Mu = np.nanmean(X, axis=0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows,].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis=0))

    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc**2).reshape(nc, nc)
            if set(O[i,]) != set(one_to_nc - 1):  # missing component exists
                M_i, O_i = M[i,][M[i,] != -1], O[i,][O[i,] != -1]  # M_i, O_i
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                # @ stands for matrix multiplication
                Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (
                    X_tilde[i, O_i] - Mu[np.ix_(O_i)]
                )
                # Imputation
                X_tilde[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        # New mu and variances
        Mu_new = np.mean(X_tilde, axis=0)
        S_new = np.cov(X_tilde.T, bias=1) + reduce(np.add, S_tilde.values()) / nr
        # Check convergence
        no_conv = (
            np.linalg.norm(Mu - Mu_new) >= eps
            or np.linalg.norm(S - S_new, ord=2) >= eps
        )
        Mu = Mu_new
        S = S_new
        iteration += 1
    result_df = pd.DataFrame(X_tilde, columns=cols)

    result = {
        "mu": Mu,
        "Sigma": S,
        "X_imputed": result_df,
        "C": C,
        "iteration": iteration,
    }

    return result


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
