import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import List, Optional

def get_word_frequency_table(corpus: list, *args, **kargs) -> pd.DataFrame:

    vectorizer = CountVectorizer(*args, **kargs )

    # sparse matrix 
    bag_of_words = vectorizer.fit_transform(corpus)

    # doc x word matrix
    # #docs, #words = X.shape
    X = bag_of_words.toarray()

    word_freq = X.sum(axis=0)

    word_freq_df = pd.DataFrame({'word_freq': word_freq})
    words = vectorizer.get_feature_names()

    word_freq_df['word'] = word_freq_df.apply(lambda r: words[r.name],axis=1)

    # sort by frequency of word
    word_freq_df.sort_values(by='word_freq',ascending=False, inplace=True)

    return word_freq_df

def get_word_tfidf_table(corpus: list, *args, **kargs) -> np.ndarray:
    
    pass


def clean_text_data(sentences: str, large_words: int , my_stopped_words = Optional[List[str]]) -> str:

    pass
