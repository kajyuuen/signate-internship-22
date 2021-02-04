import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMModel
import optuna.integration.lightgbm as optuna_lgb
from sklearn.model_selection import train_test_split

from src.utils import setup_logger


class MyLGBMModel:
    def __init__(self, name, params, fold, metrics, seeds=None, log_path=None):
        self.name = name
        self.params = params
        self.metrics = metrics
        self.kfold = fold
        self.oof = None
        self.preds = None
        self.seeds = seeds if seeds is not None else [2020]
        self.models = {}
        global logger
        logger = setup_logger(log_path)

    def build_model(self):
        model = LGBMModel(**self.params)
        return model

    def predict_cv(self, train_x, train_y, importance_name=None):
        feature_importance_df = pd.DataFrame()
        oof_seeds = []
        scores_seeds = []
        for seed in self.seeds:
            oof = []
            va_idxes = []
            scores = []
            train_x_values = train_x.values
            train_y_values = train_y.values
            fold_idx = self.kfold(train_x, train_y, random_state=seed) 

            # train and predict by cv folds
            for cv_num, (tr_idx, va_idx) in enumerate(fold_idx):
                tr_x, va_x = train_x_values[tr_idx], train_x_values[va_idx]
                tr_y, va_y = train_y_values[tr_idx], train_y_values[va_idx]
                va_idxes.append(va_idx)
                model = self.build_model()
                # fitting - train
                model.fit(tr_x, tr_y,
                          eval_set=[[va_x, va_y]],
                          early_stopping_rounds=100,
                          verbose=100)
                model_name = f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl"
                self.models[model_name] = model  # save model
                
                # predict - validation
                pred = model.predict(va_x)
                oof.append(pred)

                # validation score
                score = self.get_score(va_y, pred)
                scores.append(score)
                logger.info(f"SEED:{seed}, FOLD:{cv_num} =====> val_score:{score}")

                # visualize feature importance
                _df = pd.DataFrame()
                _df['feature_importance'] = model.feature_importances_
                _df['column'] = train_x.columns
                _df['fold'] = cv_num + 1
                feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

            # sort as default
            va_idxes = np.concatenate(va_idxes)
            oof = np.concatenate(oof)
            order = np.argsort(va_idxes)
            oof = oof[order]
            oof_seeds.append(oof)
            scores_seeds.append(np.mean(scores))
            
            # visualize feature importance
            order = feature_importance_df.groupby('column') \
                .sum()[['feature_importance']] \
                .sort_values('feature_importance', ascending=False).index[:50]
            fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
            sns.boxenplot(data=feature_importance_df, y='column', x='feature_importance', order=order, ax=ax,
                        palette='viridis')
            fig.tight_layout()
            ax.grid()
            ax.set_title('feature importance')
            fig.tight_layout()
            if importance_name is not None:
                plt.savefig(importance_name + "_seed_{}.png".format(seed))

            
        oof = np.mean(oof_seeds, axis=0)
        self.oof = oof
        logger.info(f"model:{self.name} score:{self.get_score(train_y, oof)}\n")
        return oof

    def inference(self, test_x, folds=5):
        preds_seeds = []
        for seed in self.seeds:
            preds = []
            test_x_values = test_x.values
            # train and predict by cv folds
            for cv_num in range(folds):
                print(f"-INFERENCE- SEED:{seed}, FOLD:{cv_num}")
                # load model
                model_name = f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl"
                model = self.models[model_name]
                # predict - test data
                pred = model.predict(test_x_values)
                preds.append(pred)
            preds = np.mean(preds, axis=0)
            preds_seeds.append(preds)
        preds = np.mean(preds_seeds, axis=0)
        self.preds = preds
        return preds

    def tree_importance(self, train_x, train_y, save_path=None):
        # visualize feature importance
        feature_importance_df = pd.DataFrame()
        for i, (tr_idx, va_idx) in enumerate(self.kfold(train_x, train_y)):
            tr_x, va_x = train_x.values[tr_idx], train_x.values[va_idx]
            tr_y, va_y = train_y.values[tr_idx], train_y.values[va_idx]
            model = self.build_model()
            model.fit(tr_x, tr_y,
                      eval_set=[[va_x, va_y]],
                      early_stopping_rounds=100,
                      verbose=100)
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)
        order = feature_importance_df.groupby('column') \
                    .sum()[['feature_importance']] \
                    .sort_values('feature_importance', ascending=False).index[:50]
        fig, ax = plt.subplots(figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df, y='column', x='feature_importance', order=order, ax=ax,
                      palette='viridis')
        fig.tight_layout()
        ax.grid()
        ax.set_title('feature importance')
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        return fig, feature_importance_df
    
    def find_hyper_params(self, train_x, train_y):
        best_params, history = {}, []
        tr_x, va_x, tr_y, va_y = train_test_split(train_x, train_y, test_size=0.2)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
        }
        trains = optuna_lgb.Dataset(tr_x, tr_y)
        valids = optuna_lgb.Dataset(va_x, va_y)
        model = optuna_lgb.train(params, trains, valid_sets=valids,
                                 verbose_eval=False,
                                 num_boost_round=10000,
                                 early_stopping_rounds=100)
        print(model.params)
        return model.params

    def get_score(self, y_true, y_pred):
        score = self.metrics(y_true, y_pred)
        return score