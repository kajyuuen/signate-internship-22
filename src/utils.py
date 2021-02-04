import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold

from nyaggle.feature_store import load_features, load_feature

import hydra
from scipy.optimize import minimize

from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = FileHandler(log_folder) #fh = file handler
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger

# 最適な閾値を求める関数
def threshold_optimization(y_true, y_pred, metrics=None):
    def f1_opt(x):
        if metrics is not None:
            score = -metrics(y_true, y_pred >= x)
        else:
            raise NotImplementedError
        return score
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    best_threshold = result['x'].item()
    return best_threshold

# 後で定義するモデルは確率で結果を出力するので、そこから最適なf1をgetする関数を定義
def optimized_f1(y_true, y_pred):
    bt = threshold_optimization(y_true, y_pred, metrics=f1_score)
    score = f1_score(y_true, y_pred >= bt)
    return score

def simple_f1(y_true, y_pred):
    score = f1_score(y_true, np.round(y_pred))
    return score

def lgb_f1_score(y_true, y_pred):
    y_hat = np.round(y_pred)
    return 'f1', f1_score(y_true, y_hat), True

def data_setup(is_test, feature_names):
    all_df = load_feature("all", directory=hydra.utils.to_absolute_path("features"))

    data = load_features(
        all_df,
        feature_names = feature_names,
        directory=hydra.utils.to_absolute_path("features")
    )

    train_df = data[data.state.notnull()].copy()
    test_df = data[data.state.isnull()].copy()

    if is_test:
        train_df = train_df[:200]
        test_df = test_df[:200]

    return train_df, test_df

def make_skf(train_x, train_y, folds=5, random_state=2021):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    folds_idx = [(t, v) for (t, v) in skf.split(train_x, train_y)]
    return folds_idx
