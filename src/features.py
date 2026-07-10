import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def get_tfidf_features(
    train_texts,
    test_texts,
    word_max: int = 80_000,
    char_max: int = 80_000,
):
    """
    Word-level TF-IDF + Char-level TF-IDF birləşdirir.
    Çoxdilli mətnlər üçün char n-gram daha effektivdir.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=word_max,
        sublinear_tf=True,
        min_df=2,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=char_max,
        sublinear_tf=True,
        min_df=2,
    )

    X_train_word = word_vec.fit_transform(train_texts)
    X_test_word  = word_vec.transform(test_texts)

    X_train_char = char_vec.fit_transform(train_texts)
    X_test_char  = char_vec.transform(test_texts)

    X_train = hstack([X_train_word, X_train_char])
    X_test  = hstack([X_test_word,  X_test_char])

    return X_train, X_test
