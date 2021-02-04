import numpy as np
import pandas as pd

import category_encoders as ce

from nyaggle.feature_store import cached_feature

@cached_feature("all", "features")
def create_all_df(train, test):
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    return all_df

@cached_feature("goal", "features")
def create_goal_feature(df):
    """
        調達目標金額に関する特徴量
    """
    feat = pd.DataFrame(index=df.index)
    # replace
    df["goal"] = df["goal"].replace("100000+", "100000-100000")

    # upper_goal, lower_goal
    feat["lower_goal"] = df["goal"].str.split('-', expand=True)[0].astype(float)
    feat["upper_goal"] = df["goal"].str.split('-', expand=True)[1].astype(float)

    # diff
    feat["diff_goal"] = feat["upper_goal"] - feat['lower_goal']
    feat["rate_goal"] = feat['lower_goal'] / feat["upper_goal"]

    # upper and lower flag
    feat["goal_upper_flag"] = feat["upper_goal"] == 100000
    feat["goal_lower_flag"] = feat["upper_goal"] == 1

    # bin
    feat["goal_mean"] = feat[["lower_goal", "upper_goal"]].mean(axis=1)
    feat["goal_q25"] = feat[["lower_goal", "upper_goal"]].quantile(q=0.25, axis=1)
    feat["goal_q75"] = feat[["lower_goal", "upper_goal"]].quantile(q=0.75, axis=1)

    return feat.iloc[:, -len(feat.columns):]

@cached_feature("bins", "features")
def create_bins(df, goal_feats):
    df = pd.concat([df, goal_feats], axis=1)
    feat = pd.DataFrame(index=df.index)
    df["bins_duration"] = pd.cut(df["duration"],
                                 bins=[-1, 30, 45, 60, 100],
                                 labels=['bins_d1', 'bins_d2', 'bins_d3', 'bins_d4']).astype(str)
    df["bins_goal"] = pd.cut(df["upper_goal"],
                             bins=[-1, 19999, 49999, 79999, 99999, np.inf],
                             labels=['bins_g1', 'bins_g2', 'bins_g3', 'bins_g4', 'bins_g5']).astype(str)

    feat["bins_duration"] = df["bins_duration"]
    feat["bins_goal"] = df["bins_goal"]

    # Label encoding
    category_columns = ["bins_duration", "bins_goal"]
    for c in category_columns:
        encoder = ce.OrdinalEncoder()
        feat["LE_" + c] = encoder.fit_transform(df[c])

    values = df[category_columns[0]].map(str) + "_" + df[category_columns[1]].map(str)
    encoder = ce.OrdinalEncoder()
    feat["LE_" + "-".join([category_columns[0], category_columns[1]])] = encoder.fit_transform(values)

    # Count encoding
    for c in category_columns:
        encoder = ce.CountEncoder()
        feat["CE_" + c] = encoder.fit_transform(df[c])

    values = df[category_columns[0]].map(str) + "_" + df[category_columns[1]].map(str)
    encoder = ce.CountEncoder()
    feat["CE_" + "-".join([category_columns[0], category_columns[1]])] = encoder.fit_transform(values)

    return feat.iloc[:, -len(feat.columns):]

@cached_feature("goal_cross_duration", "features")
def create_goal_cross_duration_feature(df, goal_feat):
    df = pd.concat([df, goal_feat], axis=1)
    feat = pd.DataFrame(index=df.index)

    feat["ratio_goalMax_duration"] = df["upper_goal"] / (df["duration"] + 1)
    feat["ratio_goalMin_duration"] = df["lower_goal"] / (df["duration"] + 1)
    feat["ratio_goalMean_duration"] = df["goal_mean"] / (df["duration"] + 1)
    feat["prod_goalMax_duration"] = df["upper_goal"] * (df["duration"])
    feat["prod_goalMin_duration"] = df["lower_goal"] * (df["duration"])
    feat["prod_goalMean_duration"] = df["goal_mean"] * (df["duration"])
    return feat.iloc[:, -len(feat.columns):]
