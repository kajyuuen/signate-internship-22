from nyaggle.feature_store import cached_feature

import pandas as pd
import texthero as hero

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# text の基本的な情報をgetする関数
def basic_text_features_transformer(input_df, text_columns, cleansing_hero=None, name=""):
    def _get_features(dataframe, column):
        _df = pd.DataFrame()
        _df[column + name + '_num_chars'] = dataframe[column].apply(len)
        _df[column + name + '_num_exclamation_marks'] = dataframe[column].apply(lambda x: x.count('!'))
        _df[column + name + '_num_question_marks'] = dataframe[column].apply(lambda x: x.count('?'))
        _df[column + name + '_num_punctuation'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '.,;:'))
        _df[column + name + '_num_symbols'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '*&$%'))
        _df[column + name + '_num_words'] = dataframe[column].apply(lambda x: len(x.split()))
        _df[column + name + '_num_unique_words'] = dataframe[column].apply(lambda x: len(set(w for w in x.split())))
        _df[column + name + '_words_vs_unique'] = _df[column + name + '_num_unique_words'] / _df[column + name + '_num_words']
        _df[column + name + '_words_vs_chars'] = _df[column + name + '_num_words'] / _df[column + name + '_num_chars']
        return _df
    
    # main の処理
    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].astype(str).fillna('missing')
    for c in text_columns:
        if cleansing_hero is not None:
            output_df[c] = cleansing_hero(output_df, c)
        output_df = _get_features(output_df, c)
    return output_df

def cleansing_hero_remove_html_tags(input_df, text_col):
    ## only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

def cleansing_hero_only_text(input_df, text_col):
    ## get only text (remove html tags, punctuation & digits)
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords,
        hero.preprocessing.remove_whitespace,
        hero.preprocessing.stem
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

def text_vectorizer(input_df, 
                    text_columns,
                    cleansing_hero=None,
                    vectorizer=CountVectorizer(),
                    transformer=TruncatedSVD(n_components=128),
                    name='html_count_svd'):
    
    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].astype(str).fillna('missing')
    features = []
    for c in text_columns:
        if cleansing_hero is not None:
            output_df[c] = cleansing_hero(output_df, c)

        sentence = vectorizer.fit_transform(output_df[c])
        feature = transformer.fit_transform(sentence)
        num_p = feature.shape[1]
        feature = pd.DataFrame(feature, columns=[name+str(num_p) + f'_{i:03}' for i in range(num_p)])
        features.append(feature)
    output_df = pd.concat(features, axis=1)
    return output_df

@cached_feature("raw_text_features", "features")
def create_raw_text_feature(df):
    feats = basic_text_features_transformer(input_df=df,
                                            text_columns=["html_content"],
                                            cleansing_hero=None)
    return feats

@cached_feature("remove_html_text_features", "features")
def create_remove_html_text_feature(df):
    feats = basic_text_features_transformer(input_df=df,
                                            text_columns=["html_content"],
                                            cleansing_hero=cleansing_hero_remove_html_tags,
                                            name="remove_html")
    return feats

@cached_feature("diff_text_features", "features")
def create_diff_text_feature(df, raw_text_feats, remove_text_feats):
    feat = pd.DataFrame(index=df.index)
    for raw_c, remove_c in zip(raw_text_feats.columns, remove_text_feats.columns):
        feat["diff_{}".format(raw_c)] = raw_text_feats[raw_c] - remove_text_feats[remove_c]
        feat["pre_{}".format(raw_c)] = raw_text_feats[raw_c] / (remove_text_feats[remove_c] + 10e-7)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("raw_tfidf_sdv64", "features")
def get_text_vector_raw_tfidf_sdv64(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                cleansing_hero=None,
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=64, random_state=2021),
                                name="raw_html_tfidf_sdv"
                                )
    return output_df

@cached_feature("removed_html_tags_tfidf_sdv64", "features")
def get_text_vector_removed_html_tags_tfidf_sdv64(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                cleansing_hero=cleansing_hero_remove_html_tags,  # html tag 除去の hero
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=64, random_state=2021),
                                name="removed_tags_html_tfidf_sdv"
                                )
    return output_df

@cached_feature("only_text_tfidf_sdv64", "features")
def get_text_vector_only_text_tfidf_sdv64(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                vectorizer=TfidfVectorizer(),
                                cleansing_hero=cleansing_hero_only_text,  # hero
                                transformer=TruncatedSVD(n_components=64, random_state=2021),
                                name="only_text_html_tfidf_sdv"
                                )
    return output_df

@cached_feature("raw_tfidf_sdv32", "features")
def get_text_vector_raw_tfidf_sdv32(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                cleansing_hero=None,
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=32, random_state=2021),
                                name="raw_html_tfidf_sdv"
                                )
    return output_df

@cached_feature("removed_html_tags_tfidf_sdv32", "features")
def get_text_vector_removed_html_tags_tfidf_sdv32(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                cleansing_hero=cleansing_hero_remove_html_tags,  # html tag 除去の hero
                                vectorizer=TfidfVectorizer(),
                                transformer=TruncatedSVD(n_components=32, random_state=2021),
                                name="removed_tags_html_tfidf_sdv"
                                )
    return output_df

@cached_feature("only_text_tfidf_sdv32", "features")
def get_text_vector_only_text_tfidf_sdv32(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                vectorizer=TfidfVectorizer(),
                                cleansing_hero=cleansing_hero_only_text,  # hero
                                transformer=TruncatedSVD(n_components=32, random_state=2021),
                                name="only_text_html_tfidf_sdv"
                                )
    return output_df

@cached_feature("raw_bof_nmf64", "features")
def get_text_vector_raw_bof_nmf64(df):
    output_df = text_vectorizer(df,
                                ["html_content"],
                                cleansing_hero=None,
                                vectorizer=CountVectorizer(),
                                transformer=NMF(n_components=64, random_state=2021),
                                name="raw_bof_nmf64"
                                )
    return output_df
