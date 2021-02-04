import pandas as pd
from src.features.bert import OriginalBertSentenceVectorizer

from nyaggle.feature_store import cached_feature
import texthero as hero
from transformers import AutoModel, AutoTokenizer

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

@cached_feature("get_sentence", "features")
def get_sentence(df):
    new_df = pd.DataFrame(cleansing_using_bert(df, "html_content"))
    return new_df.rename(columns={'html_content': 'cleaning_text'})

@cached_feature("bert_vec", "features")
def get_bert_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'cleaning_text': 'bert_vec'})
    bv = OriginalBertSentenceVectorizer(text_columns=["bert_vec"],
                                        use_cuda=True,
                                        truncation=True)

    text_vector = bv.fit_transform(df[["bert_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]

@cached_feature("bert_svd_64_vec", "features")
def get_bert_svd_64_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'cleaning_text': 'bert_svd_64_vec'})
    bv = OriginalBertSentenceVectorizer(text_columns=["bert_svd_64_vec"],
                                        use_cuda=True,
                                        truncation=True,
                                        n_components=64)

    text_vector = bv.fit_transform(df[["bert_svd_64_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]

@cached_feature("html_bert_vec", "features")
def get_html_bert_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'html_content': 'html_bert_vec'})
    bv = OriginalBertSentenceVectorizer(text_columns=["html_bert_vec"],
                                        use_cuda=True,
                                        truncation=True)

    text_vector = bv.fit_transform(df[["html_bert_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]

@cached_feature("html_bert_svd_64_vec", "features")
def get_html_bert_svd_64_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'html_content': 'html_bert_svd_64_vec'})
    bv = OriginalBertSentenceVectorizer(text_columns=["html_bert_svd_64_vec"],
                                        use_cuda=True,
                                        truncation=True,
                                        n_components=64)

    text_vector = bv.fit_transform(df[["html_bert_svd_64_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]

@cached_feature("bert_reduce_max_vec", "features")
def get_bert_reduce_max_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'cleaning_text': 'bert_reduce_max_vec'})
    bv = OriginalBertSentenceVectorizer(text_columns=["bert_reduce_max_vec"],
                                        pooling_strategy="reduce_max",
                                        use_cuda=True,
                                        truncation=True)

    text_vector = bv.fit_transform(df[["bert_reduce_max_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]

@cached_feature("bert_cls_token_vec", "features")
def get_bert_cls_token_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'cleaning_text': 'bert_cls_token_vec'})
    bv = OriginalBertSentenceVectorizer(text_columns=["bert_cls_token_vec"],
                                        pooling_strategy="cls_token",
                                        use_cuda=True,
                                        truncation=True)

    text_vector = bv.fit_transform(df[["bert_cls_token_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]


# Robelta
@cached_feature("html_roberta_vec", "features")
def get_html_roberta_vec(df, sentence_df):
    df = pd.concat([df, sentence_df], axis=1).rename(columns={'html_content': 'html_roberta_vec'})
    model_name = "roberta-base"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bv = OriginalBertSentenceVectorizer(text_columns=["html_roberta_vec"],
                                        model=model,
                                        tokenizer=tokenizer,
                                        use_cuda=True,
                                        truncation=True)

    text_vector = bv.fit_transform(df[["html_roberta_vec"]])
    return text_vector.iloc[:, -len(text_vector.columns):]


