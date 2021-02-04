import random
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

import hydra

from src.utils import make_skf, optimized_f1, simple_f1
from src.my_lgbm_model import MyLGBMModel
from src.my_catboost_model import MyCatBoostModel

from omegaconf import OmegaConf

from src.utils import setup_logger

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    train_df = pd.read_feather(hydra.utils.to_absolute_path("./tmp/train.f"))

    X_train = train_df.drop(cfg.drop_cols + [cfg.target_col], axis=1)
    y_train = train_df[cfg.target_col]

    
    with open(hydra.utils.to_absolute_path(cfg.save_dir + "/feature_columns.log"), mode='w') as f:
        f.write("\n".join(list(X_train.columns)))

    if cfg.models.name == "lgbm":
        model = MyLGBMModel(name = cfg.current_time, 
                            params = cfg.models.params,
                            fold = make_skf,
                            metrics = optimized_f1,
                            seeds = cfg.seeds,
                            log_path=hydra.utils.to_absolute_path(cfg.save_dir + "/train.log"))
    elif cfg.models.name == "cat":
        category_cols = [ i for i, c in enumerate(X_train.columns) if c in cfg.category_cols ]
        model = MyCatBoostModel(name = cfg.current_time, 
                                params = cfg.models.params,
                                category_cols = category_cols,
                                fold = make_skf,
                                metrics = simple_f1,
                                seeds = cfg.seeds,
                                log_path=hydra.utils.to_absolute_path(cfg.save_dir + "/train.log"))
    # Searching hyper params
    # model.find_hyper_params(X_train, y_train)

    oof = model.predict_cv(X_train, y_train, hydra.utils.to_absolute_path(cfg.save_dir + "/importance"))
    
    oof_df = pd.DataFrame(index=train_df.index)
    oof_df["pred"] = oof
    oof_df.to_csv(
        hydra.utils.to_absolute_path(cfg.save_dir + "/train_pred.csv"), header=False
    )

    with open(hydra.utils.to_absolute_path(cfg.save_dir + '/model.pickle'), 'wb') as f:
        pickle.dump(model, f)

    with open(hydra.utils.to_absolute_path(cfg.save_dir + '/config.yaml'), 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile.name)

if __name__ == "__main__":
    main()