import numpy as np
import pandas as pd
from functools import reduce
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def generate_stack_prediction(X_train, y_train, X_test, y_test):
    catboost_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=False, random_state=42)
    xgboost_model = XGBClassifier(objective='multi:softmax', num_class=6, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    catboost_model.fit(X_train, y_train)
    xgboost_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # 生成預測機率
    prob_catboost = catboost_model.predict_proba(X_test)
    prob_xgboost = xgboost_model.predict_proba(X_test)
    prob_rf = rf_model.predict_proba(X_test)

    # Stacking model使用機率當feature
    stacked_features = np.column_stack((prob_catboost, prob_xgboost, prob_rf))
    stacking_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    stacking_model.fit(stacked_features, y_test)

    y_pred_stack = stacking_model.predict(stacked_features)
    accuracy_stack = accuracy_score(y_test, y_pred_stack)
    f1_stack = f1_score(y_test, y_pred_stack, average='macro')

    print("Stacking Model Test Accuracy:", accuracy_stack)
    print("Stacking Model Test F1 Score:", f1_stack)
    
    return accuracy_stack, f1_stack