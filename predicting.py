import random
import pickle

import numpy as np
import pandas as pd

import hydra
from sklearn.metrics import f1_score

from src.utils import threshold_optimization

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    # import IPython
    # IPython.embed()
    with open(hydra.utils.to_absolute_path(cfg.save_dir + '/model.pickle'), 'rb') as f:
        model = pickle.load(f)

    train_df = pd.read_feather(hydra.utils.to_absolute_path("./tmp/train.f"))
    test_df = pd.read_feather(hydra.utils.to_absolute_path("./tmp/test.f"))
    y_pred = pd.read_csv(hydra.utils.to_absolute_path(cfg.save_dir + '/train_pred.csv'), header=None)[1].values

    y_train = train_df[cfg.target_col]
    X_test = test_df.drop(cfg.drop_cols + [cfg.target_col], axis=1)

    best_threshold = threshold_optimization(y_true=y_train.values, y_pred=y_pred, metrics=f1_score) 
    print(best_threshold)

    preds = model.inference(X_test)
    result = preds >= best_threshold

    # Result
    sample_df = pd.read_csv(hydra.utils.to_absolute_path("./data/sample_submit.csv"), header=None)
    sample_df[1] = result
    sample_df[1] = sample_df[1].astype(int)

    sample_df.to_csv(
        hydra.utils.to_absolute_path(cfg.save_dir + "/result_{}.csv".format(cfg.current_time)), header=False, index=False
    )

    # Preds
    sample_df = pd.read_csv(hydra.utils.to_absolute_path("./data/sample_submit.csv"), header=None)
    sample_df[1] = preds

    sample_df.to_csv(
        hydra.utils.to_absolute_path(cfg.save_dir + "/prob_{}.csv".format(cfg.current_time)), header=False, index=False
    )

if __name__ == "__main__":
    main()