import sys
sys.path.append("../..")

import os
from pathlib import Path

import hydra
import pandas as pd

from nyaggle.feature_store import load_features, load_feature

from src.features.aggregation_features import *
from src.features.base_features import *
from src.features.bert_features import *
from src.features.by_features import *
from src.features.html_counter_features import *
from src.features.text_features import *
from src.features.xx_encode_features import *
from src.features.scdv_features import *
from src.features.word_emb_features import *

def create_features(all_df):
    # Label, Count and Target encoding
    create_label_encode_feature(all_df)
    create_count_encode_feature(all_df)
    goal_feats = create_goal_feature(all_df)
    bins_feats = create_bins(all_df, goal_feats)

    goal_cross_duration_feats = create_goal_cross_duration_feature(all_df, goal_feats)

    # Text features
    raw_text_feats = create_raw_text_feature(all_df)
    remove_text_feats = create_remove_html_text_feature(all_df)
    diff_text_feats = create_diff_text_feature(all_df, raw_text_feats, remove_text_feats)

    # Aggregation
    aggregation_country_features(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    aggregation_category1_features(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    aggregation_category2_features(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)

    # group by
    create_by_country_feature(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    create_by_category1_feature(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    create_by_category2_feature(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    aggregation_2_features(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    aggregation_3_features(all_df, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)

    aggregation_bins_duration_features(all_df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    aggregation_bins_goal_features(all_df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)
    aggregation_bins_duration_goal_features(all_df, bins_feats, goal_feats, goal_cross_duration_feats, raw_text_feats, remove_text_feats, diff_text_feats)

    # sdv
    get_text_vector_raw_tfidf_sdv64(all_df)
    get_text_vector_removed_html_tags_tfidf_sdv64(all_df)
    get_text_vector_only_text_tfidf_sdv64(all_df)

    get_text_vector_raw_tfidf_sdv32(all_df)
    get_text_vector_removed_html_tags_tfidf_sdv32(all_df)
    get_text_vector_only_text_tfidf_sdv32(all_df)

    # nmf
    get_text_vector_raw_bof_nmf64(all_df)

    # html
    create_html_tag_all_count_feature(all_df)

    # Bert
    sentence_df = get_sentence(all_df)
    get_bert_vec(all_df, sentence_df)
    get_bert_svd_64_vec(all_df, sentence_df)
    get_html_bert_vec(all_df, sentence_df)
    get_html_bert_svd_64_vec(all_df, sentence_df)
    ##　reduce_max
    get_bert_reduce_max_vec(all_df, sentence_df)
    get_bert_cls_token_vec(all_df, sentence_df)

    # Roberta
    get_html_roberta_vec(all_df, sentence_df)

    # SCDV
    scdv_fasttext_vec(all_df, sentence_df)
    scdv_fasttext_html_content(all_df, sentence_df)
    scdv_fasttext_html_content_cluster5(all_df, sentence_df)

    # Word emb
    roberta_base_vec_from_html_content(all_df, sentence_df)
    stsb_roberta_large_from_html_content(all_df, sentence_df)
    roberta_base_vec_from_cleaning_text(all_df, sentence_df)
    stsb_roberta_large_from_cleaning_text(all_df, sentence_df)

    glove_rnn_from_html_content(all_df, sentence_df)
    stsb_roberta_large_from_html_content2(all_df, sentence_df)

def selected_features(all_df, feature_names, directory):
    print("Load Features")
    data = load_features(
        all_df,
        feature_names = feature_names,
        directory = directory
    )
    return data

def get_train_data(data):
    train_df = data[data.state.notnull()].copy()
    return train_df

def get_test_data(data):
    test_df = data[data.state.isnull()].copy()
    return test_df

def preprocessing(feature_names):
    ROOT_DIR = hydra.utils.to_absolute_path(Path(__file__).parent)
    DATA_DIR = ROOT_DIR + "/data"

    os.chdir(ROOT_DIR)

    train = pd.read_csv(DATA_DIR + "/train.csv")
    test = pd.read_csv(DATA_DIR + "/test.csv")

    all_df = create_all_df(train, test)
    create_features(all_df)

    data = selected_features(all_df, feature_names, ROOT_DIR + "/features")
    train_df = get_train_data(data).reset_index()
    test_df = get_test_data(data).reset_index()

    print("Save train_df")
    train_df.to_feather(ROOT_DIR + "/tmp/train.f")
    print("Save test_df")
    test_df.to_feather(ROOT_DIR + "/tmp/test.f")

@hydra.main(config_path="config", config_name="config")
def main(cfg):
    preprocessing(cfg.feature_names)

if __name__ == "__main__":
    main()