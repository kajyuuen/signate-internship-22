from typing import Optional

from nyaggle.feature.base import BaseFeaturizer

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec, FastText
from gensim.utils import tokenize

class SCDVVectorizer(BaseFeaturizer):
    def __init__(self,
                 embedding_type: Optional[str] = "fasttext",
                 to_lower: Optional[bool] = True,
                 vector_dim: Optional[int] = 200,
                 num_clusters: Optional[int] = 10,
                 min_count: Optional[int] = 20,
                 window_size: Optional[int] = 20,
                 num_workers: Optional[int] = 10,
                 downsampling: Optional[float] = 1e-3,
                 emb_epochs: Optional[int] = 10,
                 gmm_epochs: Optional[int] = 50,
                 threshold_percentage: Optional[float] = 0.04):

        # Global params
        self.num_clusters = num_clusters
        self.to_lower = to_lower
        self.threshold_percentage = threshold_percentage

        # Word embedding params
        self.embedding_type = embedding_type
        self.emb_epochs = emb_epochs
        self.vector_dim = vector_dim
        self.min_count = min_count
        self.window_size = window_size
        self.num_workers = num_workers
        self.downsampling = downsampling

        # GMM params
        self.gmm_epochs = gmm_epochs

    def _train_word_embedding(self, x):
        if self.embedding_type == "fasttext":
            emb_model = FastText(size=self.vector_dim,
                                      min_count=self.min_count,
                                      window=self.window_size,
                                      workers=self.num_workers,
                                      sample=self.downsampling)
        elif self.embedding_type == "word2vec":
            emb_model = Word2Vec(size=self.vector_dim,
                                 min_count=self.min_count,
                                 window=self.window_size,
                                 workers=self.num_workers,
                                 sample=self.downsampling)
        else:
            raise ValueError
        
        x = [ list(tokenize(xi, to_lower = self.to_lower)) for xi in x ]

        emb_model.build_vocab(sentences=x)
        emb_model.train(sentences=x, total_examples=len(x), epochs=self.emb_epochs)
        return emb_model, emb_model.wv.syn0, emb_model.wv.index2word

    def _train_gmm(self, word_vectors, index2word):
        clf = GaussianMixture(n_components=self.num_clusters,
                              covariance_type="tied",
                              init_params='kmeans',
                              max_iter=self.gmm_epochs)
        clf.fit(word_vectors)

        idx_proba = clf.predict_proba(word_vectors)

        word2cluster_prob = dict(zip(index2word, idx_proba))
        return word2cluster_prob

    def _calc_idf(self, x):
        tf_vectrizer = TfidfVectorizer(dtype=np.float32, lowercase=self.to_lower)
        tf_vectrizer.fit_transform(x)
        
        words = tf_vectrizer.get_feature_names()
        idfs = tf_vectrizer._tfidf.idf_

        word2idf = { word: idf for word, idf in zip(words, idfs) }
        return word2idf

    def _get_word2cluster_vectors(self, word_embedding_model, word2cluster_prob, word2idf, num_clusters, vector_dim):
        word2cluster_vectors = {}

        for word in word2cluster_prob.keys():
            try:
                word2cluster_vectors[word] = np.zeros(num_clusters * vector_dim, dtype=np.float32)
                word_vector = word_embedding_model[word]
                word_idf = word2idf[word]
                word_cluster_prob = word2cluster_prob[word]
                for cluster_id in range(num_clusters):
                    word2cluster_vectors[word][cluster_id * vector_dim:
                                            (cluster_id + 1) * vector_dim] = word_vector * word_idf * word_cluster_prob[cluster_id]
            except:
                continue
        return word2cluster_vectors
    
    def _calc_space_document_vectors(self, x, is_train):
        num_documents = len(x)
        sum_of_a_min, sum_of_a_max = 0, 0
        # Algorithm 1, line 12: Init document vector
        # (num_of_documents, num_clusters * vector_dim)
        document_vectors = np.zeros((num_documents, self.num_clusters * self.vector_dim), dtype=np.float32)
        # Algorithm 1, line 12-15: Calculate document vector
        for doc_index in range(num_documents):
            document_vector = self._calc_document_vector(x[doc_index])
            document_vectors[doc_index] = document_vector
            if is_train:
                sum_of_a_min += min(document_vector)
                sum_of_a_max += max(document_vector)

        # Calc avg a_min and a_max
        if is_train:
            avg_a_min = sum_of_a_min / num_documents
            avg_a_max = sum_of_a_max / num_documents
            t = (abs(avg_a_min) + abs(avg_a_max)) / 2
            self.threshold = t * self.threshold_percentage
        return document_vectors

    def _calc_document_vector(self, sentence):
        document_vector = np.zeros(self.num_clusters * self.vector_dim, dtype=np.float32)

        for word in tokenize(sentence, to_lower = self.to_lower):
            if word not in self.word2cluster_vectors:
                continue
            document_vector += self.word2cluster_vectors[word]

        norm = np.sqrt(np.einsum('...i,...i', document_vector, document_vector))
        if norm != 0:
            document_vector /= norm
        
        return document_vector
    
    def _make_sparse(self, document_vectors):
        zero_near_idx = abs(document_vectors) < self.threshold
        document_vectors[zero_near_idx] = 0
        return document_vectors

    def _calc_cluster_vectors(self, X):
        # Algorithm 1, line 1: Obtain word vector
        word_embedding_model, word_vectors, index2word = self._train_word_embedding(X)
        # Algorithm 1, line 2: Calculate idf
        word2idf = self._calc_idf(X)
        # Algorithm 1, line 3: Cluster word vectors using GMM
        # Algorithm 1, line 4: Obtain soft assignment P(c_k|w_i)
        word2cluster_prob = self._train_gmm(word_vectors, index2word)
        # Algorithm 1, line 5-10: Form Document Topic-vector
        word2cluster_vectors = self._get_word2cluster_vectors(word_embedding_model, word2cluster_prob, word2idf, self.num_clusters, self.vector_dim)
    
        self.word2cluster_prob = word2cluster_prob
        self.word2cluster_vectors = word2cluster_vectors

    def fit(self, X: pd.Series, y: pd.Series):
        assert len(X) == len(y)

        if y.isnull().sum() > 0:
            # y == null is regarded as test data
            X_ = X.copy()
            train_X = X_[~y.isnull()]
        else:
            train_X = X

        # Algorithm 1, line 1-10
        self._calc_cluster_vectors(X.tolist())

        # Algorithm 1, line 11-16
        # Calculate threshold
        self._calc_space_document_vectors(train_X.tolist(), is_train=True)

        return self

    def transform(self, X, y=None):
        # Algorithm 1, line 11-16
        # Calculate document vector
        document_vectors = self._calc_space_document_vectors(X.tolist(), is_train=True)
        # Make Sparse Composite Document Vectors
        scdv = self._make_sparse(document_vectors)
        return scdv

if __name__ == "__main__":
    pass