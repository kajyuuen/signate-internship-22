import pandas as pd
import flair, torch
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings

from nyaggle.feature_store import cached_feature
import texthero as hero

def transformer_document_embeddings(sentences, document_embeddings, columns_prefix):
    sentence_embeddings = []
    for sentence in sentences:
        s = Sentence(sentence)
        document_embeddings.embed(s)
        sentence_embeddings.append(s.embedding.tolist())
    columns = [ "{}_{}".format(columns_prefix, i) for i in range(len(sentence_embeddings[0])) ]
    return pd.DataFrame(sentence_embeddings, columns=columns)

# HTML content (BERT)
@cached_feature("roberta_base_vec_from_html_content", "features")
def roberta_base_vec_from_html_content(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "roberta_base_vec_from_html_content"
    embedding = TransformerDocumentEmbeddings("roberta-base")
    features = transformer_document_embeddings(df["html_content"], embedding, column_name)
    return features.iloc[:, -len(features.columns):]

@cached_feature("stsb_roberta_large_from_html_content", "features")
def stsb_roberta_large_from_html_content(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "stsb_roberta_large_from_html_content"
    embedding = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")
    features = transformer_document_embeddings(df["html_content"], embedding, column_name)
    return features.iloc[:, -len(features.columns):]

# Cleaning text (BERT)
@cached_feature("roberta_base_vec_from_cleaning_text", "features")
def roberta_base_vec_from_cleaning_text(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "roberta_base_vec_from_cleaning_text"
    embedding = TransformerDocumentEmbeddings("roberta-base")
    features = transformer_document_embeddings(df["cleaning_text"], embedding, column_name)
    return features.iloc[:, -len(features.columns):]

@cached_feature("stsb_roberta_large_from_cleaning_text", "features")
def stsb_roberta_large_from_cleaning_text(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "stsb_roberta_large_from_cleaning_text"
    embedding = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")
    features = transformer_document_embeddings(df["cleaning_text"], embedding, column_name)
    return features.iloc[:, -len(features.columns):]

# Cleaning text (Document RNN Embeddings)
@cached_feature("glove_rnn_from_html_content", "features")
def glove_rnn_from_html_content(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "glove_rnn_from_html_content"
    glove_embedding = WordEmbeddings('glove')
    document_embeddings = DocumentRNNEmbeddings([glove_embedding])
    features = transformer_document_embeddings(df["html_content"], document_embeddings, column_name)
    return features.iloc[:, -len(features.columns):]

@cached_feature("glove_rnn_from_cleaning_text", "features")
def glove_rnn_from_cleaning_text(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "glove_rnn_from_cleaning_text"
    glove_embedding = WordEmbeddings('glove')
    document_embeddings = DocumentRNNEmbeddings([glove_embedding])
    features = transformer_document_embeddings(df["cleaning_text"], document_embeddings, column_name)
    return features.iloc[:, -len(features.columns):]

# HTML content (BERT)
@cached_feature("stsb_roberta_large_from_html_content2", "features")
def stsb_roberta_large_from_html_content2(df, sentence_df):
    flair.device = torch.device('cpu') 
    df = pd.concat([df, sentence_df], axis=1)
    column_name = "stsb_roberta_large_from_html_content2"
    embedding = SentenceTransformerDocumentEmbeddings("stsb-roberta-large")
    features = transformer_document_embeddings(df["html_content"], embedding, column_name)
    return features.iloc[:, -len(features.columns):]
