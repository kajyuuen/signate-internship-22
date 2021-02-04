import xfeat
from nyaggle.feature_store import cached_feature
import pandas as pd
import itertools

def aggregation_features(df, group_key, group_values, agg_methods):
    output_df, cols = xfeat.aggregation(df, group_key, group_values, agg_methods)
    return output_df[cols]

@cached_feature("aggregation_country", "features")
def aggregation_country_features(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    group_key = "country"
    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    agg_methods = ["min", "max", "mean", "std", "count"] 
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("aggregation_category1", "features")
def aggregation_category1_features(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    group_key = "category1"
    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    agg_methods = ["min", "max", "mean", "std", "count"] 
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("aggregation_category2", "features")
def aggregation_category2_features(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    group_key = "category2"
    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    agg_methods = ["min", "max", "mean", "std", "count"] 
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("aggregation_2_features", "features")
def aggregation_2_features(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    feats = pd.DataFrame(index=df.index)

    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    category_columns = ["country", "category1", "category2"]

    agg_methods = ["min", "max", "mean", "std", "count"] 
    for c1, c2 in itertools.combinations(category_columns, 2):
        group_key = c1 + "_" + c2
        df[group_key] = df[c1].map(str) + "_" + df[c2].map(str)
        feat = aggregation_features(df, group_key, group_values, agg_methods)
        feats[feat.columns] = feat

    return feats.iloc[:, -len(feats.columns):]

@cached_feature("aggregation_3_features", "features")
def aggregation_3_features(df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    feats = pd.DataFrame(index=df.index)

    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    category_columns = ["country", "category1", "category2"]

    agg_methods = ["min", "max", "mean", "std", "count"] 
    group_key = category_columns[0] + "_" + category_columns[1] + "_" + category_columns[2]
    df[group_key] = df[category_columns[0]].map(str) + "_" + df[category_columns[1]].map(str) + "_" + df[category_columns[2]].map(str)
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    feats[feat.columns] = feat

    return feats.iloc[:, -len(feats.columns):]

@cached_feature("aggregation_bins_duration_features", "features")
def aggregation_bins_duration_features(df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    group_key = "bins_duration"
    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    agg_methods = ["min", "max", "mean", "std", "count"] 
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("aggregation_bins_goal_features", "features")
def aggregation_bins_goal_features(df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    group_key = "bins_goal"
    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    agg_methods = ["min", "max", "mean", "std", "count"] 
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("aggregation_bins_duration_goal_features", "features")
def aggregation_bins_duration_goal_features(df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats):
    df = pd.concat([df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats], axis=1)
    group_values = [
        "lower_goal",
        "upper_goal",
        "goal_mean",
        "rate_goal",
        "diff_goal",
        "ratio_goalMax_duration",
        "ratio_goalMin_duration",
        "prod_goalMax_duration",
        "prod_goalMin_duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "html_content_num_chars",
        "html_contentremove_html_num_chars",
        "diff_html_content_num_chars"
    ]
    agg_methods = ["min", "max", "mean", "std", "count"] 
    category_columns = ["bins_duration", "bins_goal"]
    group_key = category_columns[0] + "_" + category_columns[1]
    df[group_key] = df[category_columns[0]].map(str) + "_" + df[category_columns[1]].map(str)
    feat = aggregation_features(df, group_key, group_values, agg_methods)
    return feat.iloc[:, -len(feat.columns):]
