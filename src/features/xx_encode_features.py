import itertools
from nyaggle.feature_store import cached_feature
import pandas as pd
import category_encoders as ce

@cached_feature("label_encode", "features")
def create_label_encode_feature(df):
    """
    カテゴリ変数のLabelEncoding
    """
    feat = pd.DataFrame(index=df.index)

    # 1
    category_columns = ["country", "category1", "category2"]
    for c in category_columns:
        encoder = ce.OrdinalEncoder()
        feat["LE_" + c] = encoder.fit_transform(df[c])
    
    # 2
    for c1, c2 in itertools.combinations(category_columns, 2):
        values = df[c1].map(str) + "_" + df[c2].map(str)
        encoder = ce.OrdinalEncoder()
        feat["LE_" + "-".join([c1, c2])] = encoder.fit_transform(values)

    # 3
    values = df[category_columns[0]].map(str) \
        + "_" + df[category_columns[1]].map(str) \
        + "_" + df[category_columns[2]].map(str)
    encoder = ce.OrdinalEncoder()
    feat["LE_" + "-".join(category_columns)] = encoder.fit_transform(values)
    return feat.iloc[:, -len(feat.columns):]

@cached_feature("count_encode", "features")
def create_count_encode_feature(df):
    feat = pd.DataFrame(index=df.index)

    # 1
    category_columns = ["country", "category1", "category2"]
    for c in category_columns:
        encoder = ce.CountEncoder()
        feat["CE_" + c] = encoder.fit_transform(df[c])

    # 2
    for c1, c2 in itertools.combinations(category_columns, 2):
        values = df[c1].map(str) + "_" + df[c2].map(str)
        encoder = ce.CountEncoder()
        feat["CE_" + "-".join([c1, c2])] = encoder.fit_transform(values)

    # 3
    values = df[category_columns[0]].map(str) \
        + "_" + df[category_columns[1]].map(str) \
        + "_" + df[category_columns[2]].map(str)
    encoder = ce.CountEncoder()
    feat["CE_" + "-".join(category_columns)] = encoder.fit_transform(values)
    return feat.iloc[:, -len(feat.columns):]

