import pandas as pd
from src.features.scdv_vectorizer import SCDVVectorizer

from nyaggle.feature_store import cached_feature
import texthero as hero

def cleansing_using_bert(input_df, text_col):
    ## only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_whitespace,
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

@cached_feature("scdv_fasttext_vec", "features")
def scdv_fasttext_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1)
    vectorizer = SCDVVectorizer(min_count=20, vector_dim=200, num_clusters=60)

    vectorizer.fit(df["cleaning_text"], df["state"])
    vecs = vectorizer.transform(df["cleaning_text"])

    column_name = "scdv_fasttext_cleaning"
    column_names = [ "{}_{}".format(column_name, i) for i in range(vecs.shape[1])]
    features = pd.DataFrame(vecs, columns=column_names)
    return features.iloc[:, -len(features.columns):]

@cached_feature("scdv_fasttext_html_content", "features")
def scdv_fasttext_html_content(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1)
    vectorizer = SCDVVectorizer(min_count=20, vector_dim=200, num_clusters=60)

    vectorizer.fit(df["html_content"], df["state"])
    vecs = vectorizer.transform(df["html_content"])

    column_name = "scdv_fasttext_html"
    column_names = [ "{}_{}".format(column_name, i) for i in range(vecs.shape[1])]
    features = pd.DataFrame(vecs, columns=column_names)
    return features.iloc[:, -len(features.columns):]


@cached_feature("scdv_fasttext_html_content_cluster5", "features")
def scdv_fasttext_html_content_cluster5(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1)
    vectorizer = SCDVVectorizer(min_count=20, vector_dim=200, num_clusters=5)

    vectorizer.fit(df["html_content"], df["state"])
    vecs = vectorizer.transform(df["html_content"])

    column_name = "scdv_fasttext_html_cluster5"
    column_names = [ "{}_{}".format(column_name, i) for i in range(vecs.shape[1])]
    features = pd.DataFrame(vecs, columns=column_names)
    return features.iloc[:, -len(features.columns):]

