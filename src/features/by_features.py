import xfeat
from nyaggle.feature_store import cached_feature
import pandas as pd
from nyaggle.feature.groupby import aggregation

@cached_feature("by_country", "features")
def create_by_country_feature(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    feat = pd.DataFrame(index=df.index)

    columns = [
        "lower_goal",
        "upper_goal",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "prod_goalMean_duration",
    ]

    for column in columns:
        new_df, names = aggregation(df, "country", [column], ["min", "max", "mean"])
        name = names[0]
        feat[name] = new_df[name]
        feat["by_country_agg_per_{}".format(column)] = new_df[column] / new_df[name]

    return feat.iloc[:, -len(feat.columns):]

@cached_feature("by_category1", "features")
def create_by_category1_feature(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    feat = pd.DataFrame(index=df.index)

    columns = [
        "lower_goal",
        "upper_goal",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "prod_goalMean_duration",
    ]

    for column in columns:
        new_df, names = aggregation(df, "category1", [column], ["min", "max", "mean"])
        name = names[0]
        feat[name] = new_df[name]
        feat["by_category1_agg_per_{}".format(column)] = new_df[column] / new_df[name]

    return feat.iloc[:, -len(feat.columns):]

@cached_feature("by_category2", "features")
def create_by_category2_feature(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    feat = pd.DataFrame(index=df.index)

    columns = [
        "lower_goal",
        "upper_goal",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "prod_goalMean_duration",
    ]

    for column in columns:
        new_df, names = aggregation(df, "category2", [column], ["min", "max", "mean"])
        name = names[0]
        feat[name] = new_df[name]
        feat["by_category2_agg_per_{}".format(column)] = new_df[column] / new_df[name]

    return feat.iloc[:, -len(feat.columns):]
